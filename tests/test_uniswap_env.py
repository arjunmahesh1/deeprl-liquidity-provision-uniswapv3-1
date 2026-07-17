"""Tests for the rebuilt environment.

These target the specific defects that invalidated the previous paper, so each one
should fail loudly if the old behavior ever comes back.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.data.swaps import epoch_seconds  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import (  # noqa: E402
    TICK_SPACING, UniswapV3Env, tick_to_sqrt_price,
)

Q96 = 2 ** 96


T0 = 1_620_000_000  # arbitrary epoch second, hour-aligned


def make_panel(n=400, price0=3000.0, drift=0.0, pool_L=1e19, fees_per_hour=1000.0):
    """Synthetic USDC(6)/WETH(18) panel. price = USD per WETH."""
    price = price0 * np.exp(drift * np.arange(n))
    # Raw P = token1_raw/token0_raw = WETH_raw per USDC_raw = 1e12 / usd_per_weth.
    price_raw = 1e12 / price
    sqrtP = np.sqrt(price_raw)
    tick = np.log(price_raw) / np.log(1.0001)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(T0 + np.arange(n) * 3600, unit="s", utc=True),
        "price": price,
        "price_raw": price_raw,
        "sqrt_price_x96": sqrtP * Q96,
        "liquidity": np.full(n, pool_L),
        "tick": tick,
        "volume_usd": np.full(n, fees_per_hour / 0.0005),
        "fees_usd": np.full(n, fees_per_hour),
        "n_swaps": np.full(n, 10),
        "was_imputed": np.zeros(n, bool),
        "token0_usd": np.ones(n),        # USDC
        "token1_usd": price,             # WETH
    })


def make_swaps(panel):
    """One swap per hour, carrying that hour's whole fee, at that hour's price.

    The swap traverses from the previous hour's price to this one's, which is what
    the real data encodes: a Swap event reports the state AFTER the swap. With one
    swap per hour and a flat price the interval collapses to a point, so the
    per-swap model must reproduce the hourly one exactly. That equivalence is what
    makes the two comparable in the tests below.
    """
    sqrtP = panel["sqrt_price_x96"].to_numpy(float) / Q96
    prev = np.concatenate([[sqrtP[0]], sqrtP[:-1]])
    return pd.DataFrame({
        "hour": epoch_seconds(panel["timestamp"]),
        "sqrt_lo": np.minimum(prev, sqrtP),
        "sqrt_hi": np.maximum(prev, sqrtP),
        "liquidity": panel["liquidity"].to_numpy(float),
        "fee_usd": panel["fees_usd"].to_numpy(float),
        "token0_in": np.zeros(len(panel), bool),   # token1-in: prorate in sqrtP
    })


HOLD, EXIT = 0, 1
ENTER_50 = 2  # first entry in action_widths


def make_env(panel=None, enter=True, **kw):
    """Env at reset is OUT of position. `enter=True` puts it in at the first width,
    which is the starting point most tests want."""
    kw.setdefault("fee_tier_pct", 0.05)
    kw.setdefault("action_widths", np.array([50, 200, 1000]))
    kw.setdefault("warmup", 10)
    kw.setdefault("dec0", 6)
    kw.setdefault("dec1", 18)
    panel = make_panel() if panel is None else panel
    kw.setdefault("swaps", make_swaps(panel))
    env = UniswapV3Env(panel, **kw)
    env.reset()
    if enter:
        env.step(ENTER_50)
    return env


# --------------------------------------------------------------- unit safety

def test_fee_tier_is_percent_and_rate_is_fraction():
    """The bug: one `delta` meant 0.05 AND 5%. They must now be distinct."""
    env = make_env()
    assert env.fee_tier_pct == 0.05          # percent, keys tick spacing
    assert env.fee_rate == 0.0005            # fraction, the economics
    assert env.tick_spacing == TICK_SPACING[0.05]


def test_passing_the_fraction_as_the_tier_is_rejected():
    """Previously `_fee_to_tickspacing(0.0005)` raised, so the true rate was unpassable."""
    with pytest.raises(ValueError, match="unsupported fee tier"):
        make_env(fee_tier_pct=0.0005)


# ------------------------------------------------------------ enter and exit

def test_starts_out_of_position():
    env = make_env(enter=False)
    assert not env.in_position
    _, r, _, _, info = env.step(HOLD)
    assert info["fee"] == 0.0 and info["gas"] == 0.0
    assert r == pytest.approx(0.0, abs=1e-9), "staying out must score exactly the hold benchmark"


def test_staying_out_all_window_scores_zero():
    """The benchmark: never entering is worth 0, so any policy is measured against hold."""
    env = make_env(enter=False, panel=make_panel(n=200, drift=0.005))
    total = 0.0
    for _ in range(150):
        _, r, term, trunc, _ = env.step(HOLD)
        total += r
        if term or trunc:
            break
    assert total == pytest.approx(0.0, abs=1e-6)


def test_exit_stops_fees_and_costs_gas_only():
    env = make_env()
    _, _, _, _, info = env.step(EXIT)
    assert info["gas"] > 0
    assert info["swap_cost"] == 0.0, "withdrawing returns both legs; nothing is swapped"
    assert not info["in_position"]
    _, _, _, _, nxt = env.step(HOLD)
    assert nxt["fee"] == 0.0, "no fees once out"


def test_enter_costs_gas_and_swap():
    env = make_env(enter=False)
    _, _, _, _, info = env.step(ENTER_50)
    assert info["gas"] > 0 and info["swap_cost"] > 0
    assert info["in_position"]


def test_fee_income_is_share_of_gross_fees():
    """fee == fees_usd * L/(pool_L+L) while in range: the LP is not the sole provider."""
    env = make_env(panel=make_panel(pool_L=1e19, fees_per_hour=1000.0))
    _, _, _, _, info = env.step(HOLD)
    assert info["in_range"]
    expected = 1000.0 * info["share"]
    assert info["fee"] == pytest.approx(expected, rel=1e-9)
    assert 0.0 < info["share"] < 1.0, "share must be a genuine fraction of the pool"


def test_share_shrinks_when_pool_is_deeper():
    """Doubling pool liquidity roughly halves our fee. Guards the raw-unit ratio."""
    a = make_env(panel=make_panel(pool_L=1e19))
    b = make_env(panel=make_panel(pool_L=2e19))
    _, _, _, _, ia = a.step(HOLD)
    _, _, _, _, ib = b.step(HOLD)
    assert ib["fee"] < ia["fee"]
    assert ib["fee"] == pytest.approx(ia["fee"] / 2, rel=0.02)


# ------------------------------------------------------------ position logic

def test_out_of_range_earns_nothing():
    """No swap touched the band => no fee.

    Deliberately keyed to `range_frac`, not to the hour-boundary `in_range` flag.
    Asserting "in_range is False => fee == 0" is what the superseded hourly model
    did, and it is false on real data: a swap can traverse the band and settle
    outside it within the same hour, earning a real fee for an hour that ends out
    of range.
    """
    env = make_env(panel=make_panel(drift=0.02))  # walks far above the band
    earned_while_untouched = []
    for _ in range(60):
        _, _, term, trunc, info = env.step(HOLD)
        if info["range_frac"] == 0.0:
            earned_while_untouched.append(info["fee"])
        if term or trunc:
            break
    assert earned_while_untouched, "test needs the price to leave the range"
    assert all(f == 0.0 for f in earned_while_untouched)


# ------------------------------------------------- per-swap fee attribution

def test_per_swap_matches_hourly_when_price_is_static():
    """The models must agree where the discretization cannot bite.

    One swap per hour and a flat price: the swap has no extent, so "in range at the
    hour boundary" and "in range while the swap executed" are the same statement.
    Any disagreement here is a bug in the per-swap path, not the bias it removes.
    """
    panel = make_panel(drift=0.0)
    a = make_env(panel=panel, fee_model="per_swap")
    b = make_env(panel=panel, fee_model="hourly")
    for _ in range(30):
        _, _, _, _, ia = a.step(HOLD)
        _, _, _, _, ib = b.step(HOLD)
        assert ia["fee"] == pytest.approx(ib["fee"], rel=1e-9)


def test_swap_crossing_the_band_edge_earns_a_fraction():
    """A swap half inside the band earns half the fee, not all of it and not none.

    The hourly model has only those two answers, and which one it gives depends on
    where the hour happened to end.
    """
    env = make_env(enter=False)
    j = env.i + 1
    A, B = 1.0, 2.0            # band in raw sqrt-price space
    env.sw_lo[j], env.sw_hi[j] = 1.5, 2.5   # traverses [1.5, 2.0] inside, [2.0, 2.5] out
    env.sw_liq[j], env.sw_fee[j] = 0.0, 100.0   # sole provider: share == 1
    env.sw_off[j], env.sw_off[j + 1] = j, j + 1

    fee, frac = env._fee_over_hour(j, L=1.0, sqrtA=A, sqrtB=B)
    assert frac == pytest.approx(0.5)
    assert fee == pytest.approx(50.0)


def test_fee_credited_for_an_hour_that_ends_out_of_range():
    """The defect, stated as a test: the hourly model pays zero here, and is wrong.

    The swap runs through the band and settles above it. The position was the active
    liquidity for part of that swap and earned part of the fee. Ending the hour out
    of range does not undo it.
    """
    env = make_env(enter=False)
    j = env.i + 1
    A, B = 1.0, 2.0
    env.sw_lo[j], env.sw_hi[j] = 1.0, 3.0   # starts at the band's floor, ends above it
    env.sw_liq[j], env.sw_fee[j] = 0.0, 100.0
    env.sw_off[j], env.sw_off[j + 1] = j, j + 1

    fee, frac = env._fee_over_hour(j, L=1.0, sqrtA=A, sqrtB=B)
    assert frac == pytest.approx(0.5)
    assert fee == pytest.approx(50.0)


@pytest.mark.parametrize("lo,hi", [(5.0, 6.0), (0.1, 0.5)])
def test_a_swap_that_misses_the_band_earns_nothing(lo, hi):
    """Both sides of the band: entirely above it and entirely below it."""
    env = make_env(enter=False)
    j = env.i + 1
    env.sw_lo[j], env.sw_hi[j] = lo, hi
    env.sw_liq[j], env.sw_fee[j] = 0.0, 100.0
    env.sw_off[j], env.sw_off[j + 1] = j, j + 1
    fee, frac = env._fee_over_hour(j, L=1.0, sqrtA=1.0, sqrtB=2.0)
    assert (fee, frac) == (0.0, 0.0)


def test_proration_measure_follows_the_input_leg():
    """The fee is levied on the input, so the input's measure sets the proration.

    A swap traversing sqrtP in [1, 2] against a band [1, s] with s = sqrt(2):
      token1 in: input is L*d(sqrtP),   so the band absorbs (s-1)/(2-1)      = 41.4%
      token0 in: input is L*d(1/sqrtP), so the band absorbs (1-1/s)/(1-1/2)  = 58.6%
    Prorating both in sqrt-price would credit 41.4% to each, overcharging the
    token0-in direction by 17 points on this swap. Roughly half of all swaps are
    token0-in, so the wrong measure is not a corner case.
    """
    s = np.sqrt(2.0)
    for t0in, expected in [(False, (s - 1.0) / 1.0), (True, (1.0 - 1.0 / s) / 0.5)]:
        env = make_env(enter=False)
        j = env.i + 1
        env.sw_lo[j], env.sw_hi[j] = 1.0, 2.0
        env.sw_liq[j], env.sw_fee[j] = 0.0, 100.0
        env.sw_t0in[j] = t0in
        env.sw_off[j], env.sw_off[j + 1] = j, j + 1
        fee, frac = env._fee_over_hour(j, L=1.0, sqrtA=1.0, sqrtB=s)
        assert frac == pytest.approx(expected), f"token0_in={t0in}"
        assert fee == pytest.approx(100.0 * expected)


def test_each_swap_uses_its_own_liquidity_not_the_hour_end_value():
    """Two swaps, same fee, different pool depth. The share must be evaluated per
    swap: averaging them, or taking the hour's last, silently misprices the deeper
    one."""
    env = make_env(enter=False)
    j = env.i + 1
    s = env.sw_off[j]
    assert env.sw_off[j + 1] - s == 1, "fixture is one swap per hour"
    # Widen the slice to two swaps by borrowing the next hour's slot.
    env.sw_off[j + 1] = s + 2
    env.sw_lo[s:s + 2] = 1.5
    env.sw_hi[s:s + 2] = 1.5
    env.sw_fee[s:s + 2] = 100.0
    env.sw_liq[s], env.sw_liq[s + 1] = 0.0, 3.0   # shares 1.0 and 0.25 at L=1

    fee, _ = env._fee_over_hour(j, L=1.0, sqrtA=1.0, sqrtB=2.0)
    assert fee == pytest.approx(100.0 * 1.0 + 100.0 * 0.25)


def test_narrow_band_is_penalised_by_the_hourly_model():
    """The bias this rebuild removes, measured rather than asserted.

    Over a drifting price the per-swap model must credit at least as much as the
    hourly one for the same position, because the hourly model zeroes whole hours
    the band was in range for part of. Strictly more on at least one hour, or the
    defect never existed.
    """
    panel = make_panel(n=200, drift=0.001)
    a = make_env(panel=panel, fee_model="per_swap", action_widths=np.array([50]))
    b = make_env(panel=panel, fee_model="hourly", action_widths=np.array([50]))
    fa, fb, strictly_more = 0.0, 0.0, False
    for _ in range(150):
        _, _, ta, _, ia = a.step(HOLD)
        _, _, tb, _, ib = b.step(HOLD)
        fa += ia["fee"]; fb += ib["fee"]
        strictly_more |= ia["fee"] > ib["fee"] + 1e-12
        if ta or tb:
            break
    assert fa >= fb - 1e-9, "per-swap must never credit less than the hourly model"
    assert strictly_more, "the hourly model never bit: the test proves nothing"


def test_rebalance_conserves_value_net_of_costs():
    """The bug: recentring re-derived L from xt alone and manufactured the other side."""
    env = make_env(gas_usd=0.0, slippage_frac=0.0, swap_fee_frac=0.0)
    before = env._value_usd(env.L, env.i)
    env.step(ENTER_50)  # recentre, all costs zeroed
    after = env._value_usd(env.L, env.i)
    assert after == pytest.approx(before, rel=1e-6), "free rebalance must conserve value"


def test_rebalance_costs_gas_and_swap():
    env = make_env(gas_usd=5.0)
    v0 = env._value_usd(env.L, env.i)
    _, _, _, _, info = env.step(ENTER_50)
    assert info["gas"] == 5.0
    assert info["swap_cost"] > 0
    assert env._value_usd(env.L, env.i) < v0

    hold = make_env(gas_usd=5.0)
    _, _, _, _, info_hold = hold.step(HOLD)
    assert info_hold["gas"] == 0.0, "holding must never pay gas"


def test_no_free_forced_repositioning():
    """The bug: price leaving the range silently repositioned, and charged no gas."""
    env = make_env(panel=make_panel(drift=0.02))
    lo, hi = env.tick_lower, env.tick_upper
    gas0 = env.cum_gas
    for _ in range(60):
        _, _, term, trunc, info = env.step(HOLD)
        if term or trunc:
            break
    assert (env.tick_lower, env.tick_upper) == (lo, hi), "range moved without an action"
    assert env.cum_gas == gas0, "holding charged gas"


# ------------------------------------------------------------------- reward

def test_reward_decomposes_exactly():
    env = make_env()
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r == pytest.approx(info["fee"] - info["d_il"], rel=1e-9)
        if term or trunc:
            break


@pytest.mark.parametrize("gas,swap_fee,act_every", [
    (0.0, 0.0, 0),      # never acts, no costs: the identity must hold trivially
    (5.0, 0.0005, 24),  # daily
    (5.0, 0.0005, 6),   # four times a day: costs scale with acting
    (50.0, 0.0005, 24),  # pre-Dencun gas
])
def test_reward_sums_to_fees_minus_il(gas, swap_fee, act_every):
    """The episode identity: sum(reward) == cum_fees - (IL_final - IL_at_entry).

    THE test that matters, and the one the suite was missing. Fees are paid out
    rather than reinvested, and costs come out of the position, so an LP's P&L
    against the hold basket is exactly fees minus the IL it ended with.

    The bug this catches: the reward subtracted gas and swap directly AND those
    costs had already been deducted from the position's value, so they came back
    through `d_il` on the same step. Every cost was charged twice. It was invisible
    to a per-step decomposition test, because that test asserted the formula rather
    than the economics, and invisible to a passive arm, which never pays a cost.
    The double charge scaled with how often a strategy acted, so it taxed exactly
    the active management the paper is about while leaving the passive benchmark
    untouched.
    """
    panel = make_panel(n=300, drift=0.0005)
    env = make_env(panel=panel, enter=False, gas_usd=gas, swap_fee_frac=swap_fee,
                   slippage_frac=0.0, action_widths=np.array([200]), allow_exit=False)
    il_at_entry = env.prev_il
    enter = env._a_enter0          # allow_exit=False shifts the grid down by one
    total, costs, n = 0.0, 0.0, 0
    while True:
        a = enter if (act_every and n % act_every == 0) else HOLD
        _, r, term, trunc, info = env.step(a)
        total += r
        costs += info["gas"] + info["swap_cost"]
        n += 1
        if term or trunc:
            break
    identity = env.cum_fees - (env.prev_il - il_at_entry)
    assert total == pytest.approx(identity, rel=1e-6, abs=1e-6)
    if act_every:
        assert costs > 0, "test needs the strategy to actually pay something"


def test_il_telescopes_to_total_against_hold():
    """Summed per-step IL must equal the CHANGE in total IL vs the hold basket.

    Measured from entry, not from zero: entering already moved IL by the gas and
    swap it cost, so the sum telescopes to (IL_final - IL_at_entry).
    """
    env = make_env()
    il_at_start = env.prev_il
    total = 0.0
    for _ in range(50):
        _, _, term, trunc, info = env.step(HOLD)
        total += info["d_il"]
        if term or trunc:
            break
    direct = env._hold_usd(env.i) - env._value_usd(env.L, env.i) - il_at_start
    assert total == pytest.approx(direct, rel=1e-6, abs=1e-9)


def test_flat_price_earns_fees_and_no_il():
    """No price move: pure fee accrual, IL ~ 0. Sanity on the sign convention."""
    env = make_env(panel=make_panel(drift=0.0))
    fees, ils = 0.0, 0.0
    for _ in range(30):
        _, _, term, trunc, info = env.step(HOLD)
        fees += info["fee"]; ils += info["d_il"]
        if term or trunc:
            break
    assert fees > 0
    assert abs(ils) < 1e-6 * max(env.capital_usd, 1.0)


def test_observation_is_float32():
    """MPS has no float64; the old env's float64 obs made GPU training impossible."""
    env = make_env()
    obs, _, _, _, _ = env.step(HOLD)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_tick_to_sqrt_price_matches_definition():
    for t in (-100_000.0, 0.0, 195_000.0):
        assert tick_to_sqrt_price(t) == pytest.approx(np.sqrt(1.0001 ** t), rel=1e-9)
