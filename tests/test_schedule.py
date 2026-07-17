"""Tests for the decision-schedule wrappers.

The schedule sits between the agent and the environment on every training run, so a
defect here is invisible in the environment's own tests and silently changes what the
agent learns. There were none of these tests, and the wrapper was discarding every
shaped reward.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.envs.schedule import (  # noqa: E402
    EventDriven, Periodic, agent_schedule,
)
from tests.test_uniswap_env import HOLD, make_env, make_panel, make_swaps  # noqa: E402


def wrapped(spec, panel=None, **kw):
    kw.setdefault("allow_exit", False)
    kw.setdefault("action_widths", np.array([200]))
    env = make_env(panel=panel, enter=False, **kw)
    return agent_schedule(env, spec)


# ------------------------------------------------------- the shaping channel

@pytest.mark.parametrize("spec", ["hourly", "daily", "weekly", "event_driven"])
def test_schedule_passes_the_shaped_reward_through(spec):
    """The reward the agent sees must be the SHAPED one when shaping is on.

    The bug: both wrappers accumulated `info["reward_true"]` and returned that as the
    reward, discarding what the inner env actually returned. The agent is always
    wrapped in a schedule, so no control variate ever reached it, and training on
    three different shaped rewards produced bit-identical policies. That
    bit-identical-across-conditions signature is exactly what invalidated the
    previous paper's transfer result, so it gets a test.
    """
    panel = make_panel(n=300, drift=0.0, vol=0.01, seed=3)
    a = wrapped(spec, panel=panel, reward_shaping="none")
    b = wrapped(spec, panel=panel, reward_shaping="lvr")
    diff = False
    for _ in range(40):
        _, ra, ta, _, _ = a.step(HOLD)
        _, rb, tb, _, _ = b.step(HOLD)
        diff |= abs(ra - rb) > 1e-9
        if ta or tb:
            break
    assert diff, f"{spec}: shaped and unshaped rewards are identical -> shaping discarded"


@pytest.mark.parametrize("spec", ["hourly", "daily", "weekly", "event_driven"])
def test_schedule_reports_the_true_reward_regardless_of_shaping(spec):
    """`info["reward_true"]` must stay the unshaped reward, so evaluation is unaffected
    by what training optimised. This is the other half: the shaping must reach the
    agent AND must never reach a reported number."""
    panel = make_panel(n=300, drift=0.0, vol=0.01, seed=3)
    a = wrapped(spec, panel=panel, reward_shaping="none")
    b = wrapped(spec, panel=panel, reward_shaping="lvr")
    for _ in range(40):
        _, ra, ta, _, ia = a.step(HOLD)
        _, _, tb, _, ib = b.step(HOLD)
        assert ia["reward_true"] == pytest.approx(ib["reward_true"], rel=1e-9), spec
        # unshaped: the returned reward IS the true reward, which is what `score` reads
        assert ra == pytest.approx(ia["reward_true"], rel=1e-9), spec
        if ta or tb:
            break


# ------------------------------------------------------------ the accounting

@pytest.mark.parametrize("k", [1, 24, 168])
def test_periodic_accumulates_every_hour_it_holds(k):
    """A macro-step of k hours must sum k hours of reward, not report one hour of it.
    Under-accumulating would make a slow schedule look better purely by dropping the
    hours it did not look at."""
    panel = make_panel(n=400, drift=0.0, vol=0.005, seed=5)
    slow = Periodic(make_env(panel=panel, enter=False, allow_exit=False,
                             action_widths=np.array([200])), k)
    fast = make_env(panel=panel, enter=False, allow_exit=False,
                    action_widths=np.array([200]))
    _, r_slow, _, _, _ = slow.step(HOLD)
    r_fast = 0.0
    for _ in range(k):
        _, r, term, trunc, _ = fast.step(HOLD)
        r_fast += r
        if term or trunc:
            break
    assert r_slow == pytest.approx(r_fast, rel=1e-9)


def test_event_driven_holds_while_in_range_and_wakes_when_out():
    """The schedule's whole content: consult only when the price leaves the band."""
    panel = make_panel(n=400, drift=0.004)   # walks out of the band and stays out
    env = EventDriven(make_env(panel=panel, enter=False, allow_exit=False,
                               action_widths=np.array([200])), max_wait=168)
    inner = env.unwrapped
    env.step(HOLD)
    # once out of range it must stop idling, i.e. advance one hour per macro-step
    while inner.in_position and inner.sqrtA <= inner.sqrtP[inner.i] <= inner.sqrtB:
        _, _, term, trunc, _ = env.step(HOLD)
        if term or trunc:
            pytest.skip("price never left the band")
    before = inner.i
    env.step(HOLD)
    assert inner.i == before + 1, "still idling while out of range"


def test_event_driven_respects_max_wait():
    """A band the price never leaves must not idle to the end of the episode."""
    panel = make_panel(n=400, drift=0.0)     # flat: never leaves a 200-tick band
    env = EventDriven(make_env(panel=panel, enter=False, allow_exit=False,
                               action_widths=np.array([200])), max_wait=10)
    inner = env.unwrapped
    before = inner.i
    env.step(HOLD)
    assert inner.i - before <= 11, "idled past max_wait"


def test_unknown_schedule_is_rejected():
    env = make_env(enter=False)
    with pytest.raises(ValueError, match="unknown schedule"):
        agent_schedule(env, "fortnightly")
