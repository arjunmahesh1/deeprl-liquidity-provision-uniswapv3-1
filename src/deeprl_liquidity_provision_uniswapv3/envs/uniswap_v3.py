"""Uniswap v3 concentrated-liquidity environment, rebuilt on real pool flow.

What changed from the previous environment, and why each mattered:

1. `delta` was one variable carrying two incompatible conventions: a percent that
   keyed tick spacing (0.05 -> 10) and a fraction used as the fee rate. Passing the
   true 0.0005 raised in the tick lookup, so the pools charged 5% and 30%. Split
   into `fee_tier_pct` and `fee_rate`, which can never be confused.

2. Fees came from price displacement, i.e. arbitrage flow only, on a dataset with
   no volume. Measured: fee income was ~1.5% of impermanent loss, so the reward was
   ~-IL and there was nothing to trade off. Fees are now the LP's share of the
   hour's real gross fees, earned only while in range:
       fee = fees_usd * our_L / (pool_L + our_L)
   Measured after the change: fees are 14% to 60% of IL. That share term also
   retires the sole-provider assumption in the same expression.

3. Rebalancing re-derived liquidity from `xt` alone and carried `yt` over unchanged,
   with no swap; the out-of-range branch did `xt = xt/2; yt = xt*pt`, manufacturing
   the missing side. Rebalancing is now value-conserving: the position is valued,
   a swap fee and slippage are charged, and the new liquidity is solved from the
   remaining value.

4. Forced repositioning was free, because the gas indicator read the original
   action. There is no forced repositioning now: an out-of-range position simply
   stops earning, which is what actually happens on-chain. Only a chosen rebalance
   costs gas.

5. Fees were credited on an hour-boundary range check: in range at the end of the
   hour meant the whole hour's aggregated fees, out of range meant zero. With ~250
   swaps an hour a narrow band crosses in and out repeatedly inside one hour, so
   that rule charged zero for hours we earned through, and it charged it harder the
   narrower the band. It biased against concentration, which is the strategy this
   paper is about, and the bias was large enough to move the optimum: scaling fee
   income by two moved the best passive width from a degenerate +/-170% band to a
   realistic +/-5% one. Fees are now attributed PER SWAP, against each swap's own
   price interval and its own active liquidity. The agent still decides hourly (or
   at whatever cadence its schedule imposes); only the accounting is finer.

Conventions. L and sqrt-prices are RAW (matching the pool's `liquidity` column, so
the share is a legitimate ratio); token amounts are decimal-adjusted for value;
value is USD via the panel's `token0_usd` / `token1_usd`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

TICK_SPACING = {0.01: 1, 0.05: 10, 0.30: 60, 1.00: 200}


def tick_to_sqrt_price(tick: np.ndarray | float) -> np.ndarray | float:
    """Raw sqrt price at a tick. sqrt(1.0001^t) = 1.0001^(t/2)."""
    return 1.0001 ** (np.asarray(tick, dtype=float) / 2.0)


class UniswapV3Env(gym.Env):
    """Active liquidity provision as an MDP.

    Actions are a target state, not an increment:
        0        HOLD   leave the position (or lack of one) alone. Free.
        1        EXIT   burn the position and keep the tokens. Gas only: withdrawing
                        returns both legs, so nothing is swapped.
        2..k     ENTER  hold liquidity across +/- action_widths[i] ticks around the
                        current tick. Pays gas plus a swap to reach the ratio the
                        range needs, whether entering from tokens or recentring.

    EXIT matters more than width. Within a window, fees and impermanent loss both
    scale with liquidity at a near-constant ratio, so there is no interior optimum
    and the only real decision is whether to hold a position at all. Across windows
    that ratio crosses 1 about 41% of the time, which is what makes the decision
    worth learning.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: pd.DataFrame,
        fee_tier_pct: float,
        action_widths: np.ndarray,
        dec0: int,
        dec1: int,
        swaps: pd.DataFrame | None = None,
        fee_model: str = "per_swap",
        capital_usd: float = 30_000.0,
        gas_usd: float | np.ndarray = 5.0,
        allow_exit: bool = True,
        features: str = "compact",
        shaped_reward: bool = False,      # deprecated alias for reward_shaping="shadow"
        reward_shaping: str = "none",
        swap_fee_frac: float | None = None,
        slippage_frac: float = 0.0005,
        ma_windows: tuple[int, ...] = (24, 168),
        warmup: int = 168,
    ):
        super().__init__()
        if fee_tier_pct not in TICK_SPACING:
            raise ValueError(f"unsupported fee tier {fee_tier_pct}; expected one of "
                             f"{sorted(TICK_SPACING)} (PERCENT, e.g. 0.05 for the 0.05% tier)")
        # Decimals are required, never defaulted: a wrong pair silently mis-values
        # every position, which is the class of bug this rebuild exists to kill.
        self.dec0, self.dec1 = int(dec0), int(dec1)
        self.fee_tier_pct = float(fee_tier_pct)
        self.fee_rate = self.fee_tier_pct / 100.0          # the ONLY fee fraction
        self.tick_spacing = TICK_SPACING[self.fee_tier_pct]

        # action_widths are the ENTER options; HOLD (and optionally EXIT) precede them.
        self.action_widths = np.asarray(action_widths, dtype=float)
        if (self.action_widths <= 0).any():
            raise ValueError("action_widths must all be > 0; HOLD and EXIT are implicit")
        # allow_exit=False mandates a position: the LP must quote, which is the
        # market-maker's actual constraint. The benchmark is then passive LP, not
        # holding, and the question is whether managing the range mitigates the loss.
        self.allow_exit = bool(allow_exit)
        # Reward is fee - dIL, and dIL is dominated by the price path, which no action
        # influences. Writing one step out against the hold basket shows why exactly:
        #     dIL = (hold0 - a0) * dP  -  (1/2) * a0'(P) * dP^2
        # The first term is first-order in the price move and is a martingale; only
        # the second is steerable. The noise term is far larger, so the policy
        # gradient is mostly noise and PPO either freezes on HOLD or churns.
        #
        # Two control variates, and BOTH report the true reward (info["reward_true"])
        # whatever they train on. The reported metric never changes.
        #
        # "shadow" subtracts a passive shadow position held on the SAME path. The two
        # share almost all of that IL, so differencing cancels most of it. It works,
        # but the shadow is hand-built and its width is a free parameter we chose.
        #
        # "lvr" is the principled version. Benchmark against a portfolio holding the
        # position's CURRENT token amounts for one step, re-anchored every step,
        # rather than against the basket held since entry. That benchmark's
        # first-order exposure is a0*dP, which matches the LP's own first-order term
        # exactly, so the difference is the pure second-order term: the martingale is
        # removed analytically rather than approximately. This is
        # loss-versus-rebalancing, the quantity the LVR literature built for exactly
        # this decomposition. Costs must then be charged explicitly, because
        # re-anchoring each step means a level shift in value no longer reaches the
        # reward.
        if reward_shaping not in ("none", "shadow", "lvr"):
            raise ValueError(f"reward_shaping must be none|shadow|lvr, got {reward_shaping!r}")
        self.reward_shaping = "shadow" if shaped_reward else reward_shaping
        # "legacy" reproduces the rejected paper's 13-feature state, TA-Lib
        # indicators included, so its negative RL result cannot be blamed on the
        # observation. See envs/features.py.
        if features not in ("compact", "legacy"):
            raise ValueError(f"features must be compact|legacy, got {features!r}")
        self.features = features
        # "hourly" is kept only to measure what the discretization was worth; it is
        # a known-biased model and must never be the arm a claim rests on.
        if fee_model not in ("per_swap", "hourly"):
            raise ValueError(f"fee_model must be per_swap|hourly, got {fee_model!r}")
        if fee_model == "per_swap" and swaps is None:
            raise ValueError("fee_model='per_swap' needs the swap-level frame; build it "
                             "with data/swaps.py, or pass fee_model='hourly' and accept "
                             "the bias against concentration documented in SPEC.md")
        self.fee_model = fee_model
        self._a_exit = 1 if self.allow_exit else None
        self._a_enter0 = 2 if self.allow_exit else 1
        self.action_space = spaces.Discrete(self._a_enter0 + len(self.action_widths))

        self.capital_usd = float(capital_usd)
        self.gas_usd = gas_usd
        # A rebalance swaps roughly half the position through the pool itself.
        self.swap_fee_frac = self.fee_rate if swap_fee_frac is None else float(swap_fee_frac)
        self.slippage_frac = float(slippage_frac)
        self.warmup = int(warmup)

        self._load(panel, ma_windows)
        self._load_swaps(panel, swaps)

        n_feat = 13 if self.features == "legacy" else 9 + len(ma_windows)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat,), dtype=np.float32  # float32: MPS has no float64
        )

    # ------------------------------------------------------------------ data

    def _load(self, panel: pd.DataFrame, ma_windows: tuple[int, ...]) -> None:
        df = panel.reset_index(drop=True)
        need = {"price", "sqrt_price_x96", "liquidity", "tick", "fees_usd",
                "token0_usd", "token1_usd"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"panel missing columns: {sorted(missing)}")

        self.price = df["price"].to_numpy(float)
        self.sqrtP = df["sqrt_price_x96"].to_numpy(float) / 2 ** 96
        self.pool_L = df["liquidity"].to_numpy(float)
        self.tick = df["tick"].to_numpy(float)
        self.fees_usd = df["fees_usd"].to_numpy(float)
        self.t0_usd = df["token0_usd"].to_numpy(float)
        self.t1_usd = df["token1_usd"].to_numpy(float)
        self.gas = (np.full(len(df), float(self.gas_usd))
                    if np.isscalar(self.gas_usd) else np.asarray(self.gas_usd, float))

        if self.features == "legacy":
            from .features import legacy_series
            self.leg = legacy_series(self.price)
        r = pd.Series(self.price).pct_change().fillna(0.0)
        self.ret = r.to_numpy()
        self.mas = [r.rolling(w, min_periods=1).mean().to_numpy() for w in ma_windows]
        self.vol = r.rolling(24, min_periods=2).std().fillna(0.0).to_numpy()
        self.n = len(df)
        if self.n <= self.warmup + 2:
            raise ValueError(f"panel too short: {self.n} rows for warmup {self.warmup}")

    def _load_swaps(self, panel: pd.DataFrame, swaps: pd.DataFrame | None) -> None:
        """Index the window's swaps by panel row, CSR-style.

        `sw_off[j]:sw_off[j+1]` are the swaps that executed during hour j, so a step
        reads one contiguous slice instead of searching. Offsets are built for every
        row including empty hours, so a quiet hour is an empty slice rather than a
        missing key.
        """
        if swaps is None:
            self.sw_off = None
            return
        from ..data.swaps import epoch_seconds

        p_hours = epoch_seconds(panel["timestamp"])
        h = swaps["hour"].to_numpy()
        # The env is handed a window, not the whole panel: drop swaps outside it.
        keep = (h >= p_hours[0]) & (h <= p_hours[-1])
        s = swaps.loc[keep]
        row = np.searchsorted(p_hours, s["hour"].to_numpy())
        assert (p_hours[row] == s["hour"].to_numpy()).all(), \
            "swap hours do not line up with the panel's grid"

        order = np.argsort(row, kind="stable")
        row = row[order]
        self.sw_lo = s["sqrt_lo"].to_numpy(float)[order]
        self.sw_hi = s["sqrt_hi"].to_numpy(float)[order]
        self.sw_liq = s["liquidity"].to_numpy(float)[order]
        self.sw_fee = s["fee_usd"].to_numpy(float)[order]
        self.sw_t0in = s["token0_in"].to_numpy(bool)[order]
        self.sw_off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(row, minlength=self.n), out=self.sw_off[1:])

    def _fee_over_hour(self, j: int, L: float, sqrtA: float, sqrtB: float):
        """Fees a position [sqrtA, sqrtB] of size L earns during hour j.

        Per swap, two things are true that the hourly model could not express.

        A swap has EXTENT in price, not a location: the event reports the price after
        the swap, so the swap traversed [sqrt_lo, sqrt_hi] from the previous swap's
        price. A band covering part of that interval was the active liquidity for
        only part of the swap, and earns that part of the fee. Uniswap v3 credits fee
        growth per unit of liquidity as the price crosses each initialized tick
        (`SwapMath.computeSwapStep` runs once per tick range), so the fee a position
        earns is the portion of the swap that executed inside its own range.

        The fee is levied on the INPUT leg, so the fraction of the fee a sub-interval
        carries is the fraction of the INPUT it absorbed, and which measure that is
        depends on the direction. Over a step at constant L the two legs are
            token1 in:  dy = L * d(sqrtP)          -> linear in sqrtP
            token0 in:  dx = L * d(1/sqrtP)        -> linear in 1/sqrtP
        so a single sqrt-price proration would be exact for one direction and wrong
        for the other, and roughly half of all swaps are token0-in. Each swap is
        apportioned in its own measure.

        Each swap also carries its OWN active liquidity, so the share is evaluated at
        the swap rather than at whatever L happened to be standing at the hour's end.

        Returns (fee_usd, fee_weighted_in_range_fraction).
        """
        s, e = self.sw_off[j], self.sw_off[j + 1]
        if e <= s or L <= 0:
            return 0.0, 0.0
        lo, hi = self.sw_lo[s:e], self.sw_hi[s:e]
        t0in = self.sw_t0in[s:e]
        # Overlap and extent, measured in sqrtP for token1-in swaps and in 1/sqrtP for
        # token0-in ones. Inverting flips the interval, hence the swapped endpoints.
        c_lo, c_hi = np.maximum(lo, sqrtA), np.minimum(hi, sqrtB)
        span = np.where(t0in, 1.0 / lo - 1.0 / hi, hi - lo)
        ov = np.where(t0in,
                      np.where(c_hi > 0, 1.0 / np.maximum(c_lo, 1e-300)
                               - 1.0 / np.maximum(c_hi, 1e-300), 0.0),
                      c_hi - c_lo)
        # A swap that did not move the price has no interval to apportion: it either
        # executed inside the band or it did not.
        moved = span > 0
        frac = np.where(moved,
                        np.clip(ov, 0.0, None) / np.where(moved, span, 1.0),
                        ((lo >= sqrtA) & (lo <= sqrtB)).astype(float))
        # An interval disjoint from the band inverts to a negative overlap in either
        # measure, but clip alone cannot see a band that sits entirely outside it.
        frac = np.where(c_hi >= c_lo, frac, 0.0)
        f = self.sw_fee[s:e]
        # The pool's reported liquidity excludes our hypothetical position, so our
        # share of the fee is L/(pool_L + L): adding our own liquidity dilutes us.
        share = L / (self.sw_liq[s:e] + L)
        earned = float((f * frac * share).sum())
        gross = float(f.sum())
        return earned, (float((f * frac).sum()) / gross if gross > 0 else 0.0)

    def _fee_hourly(self, j: int, L: float, sqrtA: float, sqrtB: float):
        """The superseded model: hour-boundary range check, whole hour's fees."""
        if not (sqrtA <= self.sqrtP[j] <= sqrtB) or L <= 0:
            return 0.0, 0.0
        denom = self.pool_L[j] + L
        return float(self.fees_usd[j] * (L / denom if denom > 0 else 0.0)), 1.0

    def _fee(self, j: int, L: float, sqrtA: float, sqrtB: float):
        if self.fee_model == "per_swap":
            return self._fee_over_hour(j, L, sqrtA, sqrtB)
        return self._fee_hourly(j, L, sqrtA, sqrtB)

    # ------------------------------------------------- position mathematics

    def _amounts(self, L: float, sqrtP: float, sqrtA: float, sqrtB: float):
        """Raw token amounts of a v3 position. Handles all three branches."""
        if sqrtP <= sqrtA:                       # entirely token0
            return L * (1.0 / sqrtA - 1.0 / sqrtB), 0.0
        if sqrtP >= sqrtB:                       # entirely token1
            return 0.0, L * (sqrtB - sqrtA)
        return L * (1.0 / sqrtP - 1.0 / sqrtB), L * (sqrtP - sqrtA)

    def _current_amounts(self, i: int):
        """Raw amounts we hold: from the position when in, or the loose tokens when out."""
        if not self.in_position:
            return self.out0, self.out1
        return self._amounts(self.L, self.sqrtP[i], self.sqrtA, self.sqrtB)

    def _usd(self, a0: float, a1: float, i: int) -> float:
        return a0 / 10 ** self.dec0 * self.t0_usd[i] + a1 / 10 ** self.dec1 * self.t1_usd[i]

    def _value_usd(self, L: float, i: int) -> float:
        """USD value of what we hold right now. `L` is ignored when out of position."""
        return self._usd(*self._current_amounts(i), i)

    def _liquidity_for_value(self, value_usd: float, i: int, sqrtA: float, sqrtB: float) -> float:
        """Solve L from a target USD value. Value is linear in L, so one evaluation suffices."""
        a0, a1 = self._amounts(1.0, self.sqrtP[i], sqrtA, sqrtB)
        per_L = a0 / 10 ** self.dec0 * self.t0_usd[i] + a1 / 10 ** self.dec1 * self.t1_usd[i]
        return value_usd / per_L if per_L > 0 else 0.0

    def _scale_to_value(self, a0: float, a1: float, target_usd: float, i: int):
        v = self._usd(a0, a1, i)
        k = max(target_usd, 0.0) / v if v > 0 else 0.0
        return a0 * k, a1 * k

    def _set_range(self, i: int, width: float, value_usd: float) -> None:
        centre = np.round(self.tick[i] / self.tick_spacing) * self.tick_spacing
        # The half-width snaps to the tick spacing too, not just the centre: a
        # position may only be minted on initializable ticks, so at the 0.30% tier
        # (spacing 60) a 100-tick half-width is not a position anyone could open.
        w = max(np.round(width / self.tick_spacing) * self.tick_spacing, self.tick_spacing)
        self.tick_lower, self.tick_upper = centre - w, centre + w
        self.sqrtA = tick_to_sqrt_price(self.tick_lower)
        self.sqrtB = tick_to_sqrt_price(self.tick_upper)
        self.L = self._liquidity_for_value(value_usd, i, self.sqrtA, self.sqrtB)
        self.in_position = True

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = self.warmup
        # Start out of position, holding the basket a passive holder would keep.
        # That basket is the benchmark, so staying out for a whole window scores 0.
        self.in_position = False
        self.L = 0.0
        self.sqrtA = self.sqrtB = 0.0
        self.tick_lower = self.tick_upper = 0.0
        w0 = float(self.action_widths[0])
        sqrtP0 = self.sqrtP[self.i]
        sA, sB = tick_to_sqrt_price(self.tick[self.i] - w0), tick_to_sqrt_price(self.tick[self.i] + w0)
        a0, a1 = self._amounts(1.0, sqrtP0, sA, sB)
        per_L = self._usd(a0, a1, self.i)
        L0 = self.capital_usd / per_L if per_L > 0 else 0.0
        self.out0, self.out1 = self._amounts(L0, sqrtP0, sA, sB)
        self.hold0 = self.out0 / 10 ** self.dec0
        self.hold1 = self.out1 / 10 ** self.dec1
        self.prev_il = 0.0
        self.cum_fees = 0.0
        self.cum_gas = 0.0
        self.cum_lvr = 0.0
        self.n_rebalances = 0
        self.n_exits = 0
        if not self.allow_exit:
            # Mandated to quote: enter at the first width, free of entry cost so the
            # comparison against passive LP is not contaminated by a one-off charge.
            self._set_range(self.i, float(self.action_widths[0]), self.capital_usd)
        # Passive shadow on the same path, for the control variate. Never acts.
        w_s = float(self.action_widths[0])
        c = np.round(self.tick[self.i] / self.tick_spacing) * self.tick_spacing
        self.sh_A = tick_to_sqrt_price(c - w_s)
        self.sh_B = tick_to_sqrt_price(c + w_s)
        a0s, a1s = self._amounts(1.0, self.sqrtP[self.i], self.sh_A, self.sh_B)
        per = self._usd(a0s, a1s, self.i)
        self.sh_L = self.capital_usd / per if per > 0 else 0.0
        self.sh_prev_il = 0.0
        self.prev_il = 0.0
        self.cum_fees = 0.0
        self.cum_gas = 0.0
        self.cum_lvr = 0.0
        self.n_rebalances = 0
        self.n_exits = 0
        return self._obs(), {}

    def _hold_usd(self, i: int) -> float:
        return self.hold0 * self.t0_usd[i] + self.hold1 * self.t1_usd[i]

    def _obs(self) -> np.ndarray:
        if self.features == "legacy":
            return self._obs_legacy()
        i = self.i
        if self.in_position:
            in_range = float(self.sqrtA <= self.sqrtP[i] <= self.sqrtB)
            span = self.sqrtB - self.sqrtA
            pos = float(np.clip((self.sqrtP[i] - self.sqrtA) / span, -1.0, 2.0)) if span > 0 else 0.5
            denom = self.pool_L[i] + self.L
            share = self.L / denom if denom > 0 else 0.0
            width = (self.tick_upper - self.tick_lower) / 100.0
        else:
            in_range = pos = share = width = 0.0
        # The signal the timing decision needs: what the pool is paying per dollar of
        # liquidity right now, against how much the price is moving.
        fee_rate_per_liq = self.fees_usd[i] / self.pool_L[i] if self.pool_L[i] > 0 else 0.0
        feats = [
            self.ret[i], self.vol[i], float(self.in_position), in_range, pos, share,
            width, self.fee_rate * 1000.0, np.log1p(fee_rate_per_liq * 1e18),
            *[m[i] for m in self.mas],
        ]
        return np.asarray(feats, dtype=np.float32)

    def _obs_legacy(self) -> np.ndarray:
        """The rejected paper's exact state vector, in its order."""
        i = self.i
        feats = [
            self.price[i],
            self.tick[i],
            self.tick_upper - self.tick_lower,
            self.L,
            self.vol[i],
            self.mas[0][i], self.mas[1][i],
            self.leg["bb_upper"][i], self.leg["bb_middle"][i], self.leg["bb_lower"][i],
            self.leg["adxr"][i], self.leg["bop"][i], self.leg["dx"][i],
        ]
        return np.asarray(feats, dtype=np.float32)

    def step(self, action):
        i = self.i
        a = int(action)
        gas_cost = swap_cost = 0.0

        if self.allow_exit and a == self._a_exit and self.in_position:
            # Withdraw. Both legs come back, so no swap: gas only.
            self.out0, self.out1 = self._current_amounts(i)
            self.in_position = False
            self.L = 0.0
            gas_cost = float(self.gas[i])
            v = self._usd(self.out0, self.out1, i)
            self.out0, self.out1 = self._scale_to_value(self.out0, self.out1, v - gas_cost, i)
            self.n_exits += 1

        elif a >= self._a_enter0:
            width = float(self.action_widths[a - self._a_enter0])
            v = self._value_usd(self.L, i)
            gas_cost = float(self.gas[i])
            # Reaching the ratio a range needs means swapping ~half the position,
            # whether we are entering from tokens or recentring an existing range.
            swap_cost = 0.5 * v * (self.swap_fee_frac + self.slippage_frac)
            self._set_range(i, width, max(v - gas_cost - swap_cost, 0.0))
            self.n_rebalances += 1

        self.cum_gas += gas_cost

        # The rebalancing benchmark's anchor: what we hold once the action is done.
        # Taken AFTER the action so a cost is a level shift the benchmark never sees,
        # which is why the LVR reward charges gas and swap explicitly.
        anc0, anc1 = self._current_amounts(i)

        j = i + 1
        fee = 0.0
        in_range = False
        range_frac = 0.0
        share = 0.0
        if self.in_position:
            fee, range_frac = self._fee(j, self.L, self.sqrtA, self.sqrtB)
            # Boundary state, for the observation and as a diagnostic. The fee no
            # longer depends on it: `range_frac` is the fraction of the hour's fees
            # our band was actually eligible for.
            in_range = self.sqrtA <= self.sqrtP[j] <= self.sqrtB
            denom = self.pool_L[j] + self.L
            share = self.L / denom if denom > 0 else 0.0
        self.cum_fees += fee

        # IL against the hold basket, per-step so the reward telescopes to the total.
        il_total = self._hold_usd(j) - self._value_usd(self.L, j)
        d_il = il_total - self.prev_il
        self.prev_il = il_total

        # Costs are NOT subtracted here. They are already paid: acting deducted them
        # from the position's value, so they arrive through `d_il` on this very step.
        # Subtracting them again charged every cost twice, and the double charge fell
        # only on strategies that act, which is every arm except passive. Fees are
        # paid out rather than reinvested, so the episode identity is
        #     sum(reward) == cum_fees - (IL_final - IL_at_entry)
        # and that is what `test_reward_sums_to_fees_minus_il` holds it to.
        reward = fee - d_il
        reward_true = reward

        # Loss versus rebalancing: hold the post-action amounts for one step and mark
        # them at j, against what the pool actually left us at j. The difference is
        # what arbitrageurs took, non-negative by the concavity of the AMM's value in
        # price, and free of the first-order price exposure that dominates dIL.
        lvr = self._usd(anc0, anc1, j) - self._usd(*self._current_amounts(j), j)

        if self.reward_shaping == "lvr":
            reward = fee - lvr - gas_cost - swap_cost

        elif self.reward_shaping == "shadow":
            sh_fee, _ = self._fee(j, self.sh_L, self.sh_A, self.sh_B)
            a0s, a1s = self._amounts(self.sh_L, self.sqrtP[j], self.sh_A, self.sh_B)
            sh_il = self._hold_usd(j) - self._usd(a0s, a1s, j)
            sh_d_il = sh_il - self.sh_prev_il
            self.sh_prev_il = sh_il
            reward = reward_true - (sh_fee - sh_d_il)

        self.i = j
        terminated = self.i >= self.n - 2
        truncated = False
        self.cum_lvr += lvr
        info = {
            "fee": fee, "d_il": d_il, "lvr": lvr, "gas": gas_cost, "swap_cost": swap_cost,
            "in_range": bool(in_range), "range_frac": range_frac, "share": share,
            "in_position": self.in_position,
            "cum_fees": self.cum_fees, "cum_gas": self.cum_gas, "cum_lvr": self.cum_lvr,
            "n_rebalances": self.n_rebalances, "n_exits": self.n_exits,
            "reward_true": reward_true,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
