"""The paper's four competitor strategies, ported to the rebuilt environment.

Functional forms and parameter grids follow the originals in
``baseline_strategies.py`` so the comparison stays with the paper's own
competitor set. Three things about the originals do not carry over, and each was
a defect rather than a design choice:

1. **They selected on the test window.** Each `evaluate()` swept its parameters and
   returned the argmax on the same segment it was scored on. Here a policy is a
   plain callable; parameter selection happens once on VALIDATION, in the runner,
   with the same budget PPO gets.

2. **PassiveWidthSweep swept nothing.** It filtered candidates to those present in
   `env.action_values`, and on WETH the intersection of `range(20,201,10)` with
   `[0,45,50,55]` was the single width `[50]`. The sweep is over a real grid here.

3. **Widths were snapped to a 4-element action grid** that Optuna had itself chosen
   by test reward, so the closed-form widths collapsed onto 0/45/50/55. Widths snap
   to whatever grid the run declares, and the grid is a pre-registered choice.

`base_factor` converts a volatility into ticks. A tick is 1.0001x, so a fractional
move s spans about s/1e-4 = s*1e4 ticks. The originals used 100, which understates
it by 100x; both are exposed so the discrepancy is visible rather than buried.
"""
from __future__ import annotations

import math

import numpy as np

HOLD = 0


class Policy:
    """Callable over (obs, env) -> action index, with a name for reporting."""

    name = "policy"

    def __call__(self, obs, env) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


def _nearest_width_action(env, width: float) -> int:
    """Index of the ENTER action whose width is closest to `width`."""
    i = int(np.argmin(np.abs(env.action_widths - width)))
    return env._a_enter0 + i


def _in_range(env) -> bool:
    return bool(env.in_position and env.sqrtA <= env.sqrtP[env.i] <= env.sqrtB)


class PassiveWidthSweep(Policy):
    """Deposit once at a fixed width, then hold. The width is the free parameter."""

    def __init__(self, width: float):
        self.width = float(width)
        self.name = f"PassiveWidthSweep(w={width:g})"
        self._deposited = False

    def reset(self) -> None:
        self._deposited = False

    def __call__(self, obs, env) -> int:
        if not self._deposited:
            self._deposited = True
            return _nearest_width_action(env, self.width)
        return HOLD


class VolProportionalWidth(Policy):
    """Width proportional to realized volatility: width ~ k * sigma * base_factor.

    The original recomputed and recentred every hour, which pays gas hourly. That
    is kept, since it is the strategy the paper compares against.
    """

    def __init__(self, k: float, base_factor: float = 1e4, only_when_out: bool = False):
        self.k = float(k)
        self.base_factor = float(base_factor)
        self.only_when_out = bool(only_when_out)
        self.name = f"VolProportionalWidth(k={k:g}{',out' if only_when_out else ''})"

    def __call__(self, obs, env) -> int:
        if self.only_when_out and _in_range(env):
            return HOLD
        width = self.k * env.vol[env.i] * self.base_factor
        return _nearest_width_action(env, width)


class ILMinimizer(Policy):
    """Closed form: width ~ 2 * sigma * sqrt(H) * base_factor. No free parameters.

    The original had zero tunable parameters while PPO got 10 Optuna trials plus its
    choice of action grid, so the budgets were never matched. `horizon` is exposed
    here so it can receive the same validation budget as everything else.
    """

    def __init__(self, horizon: int = 24, base_factor: float = 1e4, only_when_out: bool = False):
        self.H = int(horizon)
        self.base_factor = float(base_factor)
        self.only_when_out = bool(only_when_out)
        self.name = f"ILMinimizer(H={horizon}{',out' if only_when_out else ''})"

    def __call__(self, obs, env) -> int:
        if self.only_when_out and _in_range(env):
            return HOLD
        width = 2.0 * env.vol[env.i] * math.sqrt(self.H) * self.base_factor
        return _nearest_width_action(env, width)


class ReactiveRecentering(Policy):
    """Recentre at a fixed width when volatility or a price jump crosses a threshold."""

    def __init__(self, width: float, vol_threshold: float, jump_threshold: float):
        self.width = float(width)
        self.v_th = float(vol_threshold)
        self.j_th = float(jump_threshold)
        self.name = f"ReactiveRecentering(w={width:g},v={vol_threshold:g},j={jump_threshold:g})"
        self._last_price = None

    def reset(self) -> None:
        self._last_price = None

    def __call__(self, obs, env) -> int:
        p = env.price[env.i]
        if self._last_price is None:
            self._last_price = p
            return _nearest_width_action(env, self.width)
        jump = abs(p - self._last_price) / max(self._last_price, 1e-12)
        if env.vol[env.i] > self.v_th or jump > self.j_th:
            self._last_price = p
            return _nearest_width_action(env, self.width)
        return HOLD


class RecentreWhenOut(Policy):
    """Redeploy at a fixed width whenever the price leaves the band.

    Not one of the paper's four. It is the strongest thing found so far and the
    natural event-driven rule, so it belongs in the comparison as the baseline to
    beat rather than as a strawman.
    """

    def __init__(self, width: float):
        self.width = float(width)
        self.name = f"RecentreWhenOut(w={width:g})"

    def __call__(self, obs, env) -> int:
        if _in_range(env):
            return HOLD
        return _nearest_width_action(env, self.width)


class Passive(Policy):
    """Never act. Under a quoting mandate this is the do-nothing LP benchmark."""

    name = "Passive"

    def __call__(self, obs, env) -> int:
        return HOLD


def candidate_grid(widths) -> list[Policy]:
    """Every competitor configuration, for selection on validation.

    Grids follow the originals: k in {3,5,7,10,15}; ReactiveRecentering over a 3x3
    of volatility and jump thresholds.
    """
    out: list[Policy] = [Passive()]
    for w in widths:
        out.append(PassiveWidthSweep(w))
        out.append(RecentreWhenOut(w))
    for k in (3, 5, 7, 10, 15):
        out.append(VolProportionalWidth(k))
        out.append(VolProportionalWidth(k, only_when_out=True))
    for H in (24, 168):
        out.append(ILMinimizer(H))
        out.append(ILMinimizer(H, only_when_out=True))
    for w in widths:
        for v_th in (0.005, 0.01, 0.02):
            for j_th in (0.005, 0.01, 0.02):
                out.append(ReactiveRecentering(w, v_th, j_th))
    return out
