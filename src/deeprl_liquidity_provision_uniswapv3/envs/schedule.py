"""Decision schedules: how often a POLICY chooses to act.

The schedule is part of the strategy, not of the world. "I rebalance hourly" is a
choice a strategy makes, and forcing every policy onto one schedule destroys what
distinguishes the competitors: `ReactiveRecentering` fires on a volatility or jump
signal, `VolProportionalWidth` recomputes a width, `Passive` never acts. Each is a
complete specification including its own timing.

So the competitors run on their native logic, consulted every hour, and decide for
themselves whether to act. A learned agent has no native timing, so it needs one
declared: `Periodic(24)` is the pre-registered primary, with 1 and 168 as
sensitivity.

Measured on validation, PPO by decision frequency: hourly -6,502, daily -3,791,
weekly -1,509, event-driven -1,965. Daily is deliberately not the most favorable
choice; it is the realistic operating frequency for an LP, and a claim that holds
there is worth more than one that needs a weekly cadence to survive.

A wrapper here accumulates TWO rewards over the hours it holds, and the difference
matters. `reward` is what the inner env returned, which is what the agent trains on
and is shaped when shaping is on. `info["reward_true"]` is the unshaped reward, which
is what evaluation reports. Accumulating only `reward_true` and returning it as the
reward, as these wrappers first did, silently discarded every control variate: the
agent is always wrapped in a schedule, so no shaping ever reached it, and three
different shaped rewards trained bit-identical policies. Keep the two separate.
"""
from __future__ import annotations

import gymnasium as gym


class Periodic(gym.Wrapper):
    """Consult the policy every `k` hours; hold in between.

    k=1 reproduces the previous paper's implicit schedule: 1,500 chances per window
    to churn at roughly $20 each.
    """

    def __init__(self, env, k: int = 24):
        super().__init__(env)
        self.k = int(k)
        self.name = f"periodic({k}h)"

    def step(self, action):
        total, true_total, term, trunc, info = 0.0, 0.0, False, False, {}
        obs = None
        for t in range(self.k):
            obs, r, term, trunc, info = self.env.step(action if t == 0 else 0)
            total += r
            true_total += info["reward_true"]
            if term or trunc:
                break
        return obs, total, term, trunc, {**info, "reward_true": true_total}


class EventDriven(gym.Wrapper):
    """Consult when the price leaves the range, or after `max_wait` hours.

    Kept as a sensitivity arm. Note it collapses the competitor set's diversity if
    applied to them: a rule consulted only when out of range can never fire its own
    trigger, so `ReactiveRecentering` degenerates toward `RecentreWhenOut`.
    """

    def __init__(self, env, max_wait: int = 168):
        super().__init__(env)
        self.max_wait = int(max_wait)
        self.name = "event_driven"

    def _idle(self) -> bool:
        e = self.env.unwrapped
        return bool(e.in_position and e.sqrtA <= e.sqrtP[e.i] <= e.sqrtB)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        total, true_total = r, info["reward_true"]
        waited = 0
        while not (term or trunc) and self._idle() and waited < self.max_wait:
            obs, r, term, trunc, info = self.env.step(0)
            total += r
            true_total += info["reward_true"]
            waited += 1
        return obs, total, term, trunc, {**info, "reward_true": true_total}


def agent_schedule(env, spec: str = "daily"):
    """Wrap an env with the AGENT's declared schedule. Competitors are never wrapped."""
    table = {"hourly": lambda e: Periodic(e, 1),
             "daily": lambda e: Periodic(e, 24),
             "weekly": lambda e: Periodic(e, 168),
             "event_driven": EventDriven}
    if spec not in table:
        raise ValueError(f"unknown schedule {spec!r}; have {sorted(table)}")
    return table[spec](env)
