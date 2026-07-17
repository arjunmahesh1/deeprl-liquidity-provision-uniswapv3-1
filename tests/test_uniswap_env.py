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
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import (  # noqa: E402
    TICK_SPACING, UniswapV3Env, tick_to_sqrt_price,
)

Q96 = 2 ** 96


def make_panel(n=400, price0=3000.0, drift=0.0, pool_L=1e19, fees_per_hour=1000.0):
    """Synthetic USDC(6)/WETH(18) panel. price = USD per WETH."""
    price = price0 * np.exp(drift * np.arange(n))
    # Raw P = token1_raw/token0_raw = WETH_raw per USDC_raw = 1e12 / usd_per_weth.
    price_raw = 1e12 / price
    sqrtP = np.sqrt(price_raw)
    tick = np.log(price_raw) / np.log(1.0001)
    return pd.DataFrame({
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
    env = UniswapV3Env(panel if panel is not None else make_panel(), **kw)
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
    env = make_env(panel=make_panel(drift=0.02))  # walks far above the band
    earned_while_out = []
    for _ in range(60):
        _, _, term, trunc, info = env.step(HOLD)
        if not info["in_range"]:
            earned_while_out.append(info["fee"])
        if term or trunc:
            break
    assert earned_while_out, "test needs the price to leave the range"
    assert all(f == 0.0 for f in earned_while_out)


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
        assert r == pytest.approx(
            info["fee"] - info["d_il"] - info["gas"] - info["swap_cost"], rel=1e-9)
        if term or trunc:
            break


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
