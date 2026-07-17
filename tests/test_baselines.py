"""Tests for the ported competitor strategies.

The competitors are the thing every claim is measured against, so a defect here moves
every number in the paper. They had no tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.policies import baselines as B  # noqa: E402
from tests.test_uniswap_env import HOLD, make_panel, make_swaps  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env  # noqa: E402


def env_on(panel=None, widths=(45, 50, 55)):
    """The previous paper's configuration: its grid, its units, its defaults."""
    panel = make_panel(n=300, drift=0.0, vol=0.01, seed=4) if panel is None else panel
    e = UniswapV3Env(panel, swaps=make_swaps(panel), fee_tier_pct=0.05,
                     action_widths=np.array(widths), dec0=6, dec1=18, warmup=10)
    e.reset()
    return e


# ------------------------------------------------- the width -> action mapping

def test_a_near_zero_width_means_do_nothing():
    """The previous paper's grid is `[0, 45, 50, 55]` and its 0 means "do nothing"
    (`custom_env.py` recentres only `if action != 0`). Dropping the 0 from the candidate
    list forces every computed width, however small, onto a real band."""
    e = env_on()
    assert B._nearest_width_action(e, 0.0) == HOLD
    assert B._nearest_width_action(e, 1.0) == HOLD      # int(3*0.006*100) = 1
    assert B._nearest_width_action(e, 22.0) == HOLD     # still nearer 0 than 45


def test_a_width_on_the_grid_maps_to_that_width():
    """...and the 0 must not swallow the real widths."""
    e = env_on()
    for i, w in enumerate((45, 50, 55)):
        assert B._nearest_width_action(e, float(w)) == e._a_enter0 + i


def test_an_overshooting_width_saturates_at_the_widest():
    e = env_on()
    assert B._nearest_width_action(e, 10_000.0) == e._a_enter0 + 2


# ------------------------------------------------- what the competitors ARE
#
# Recording the previous paper's competitor set as it actually behaves. These are not
# aspirational: they are what its tables were measured against.

def test_vol_proportional_barely_acts_at_the_papers_base_factor():
    """`base_factor=100` against an hourly ew_sigma of ~0.006 computes ~1, whose nearest
    action is 0. The paper's volatility-proportional baseline is, in practice, passive.

    This is not a bug to repair. Raising base_factor so the strategy "works" would be
    inventing a competitor the paper never ran. It IS worth knowing when reading the
    paper's tables.
    """
    e = env_on()
    acts = set()
    for _ in range(200):
        a = B.VolProportionalWidth(3)(None, e)
        acts.add(a)
        _, _, term, trunc, _ = e.step(a)
        if term or trunc:
            break
    assert acts == {HOLD}, f"expected the paper's baseline to hold throughout, got {acts}"


def test_base_factor_is_what_makes_it_act():
    """The knob the original's own comment calls "adjust empirically" and never adjusts.
    Exposed so a non-degenerate version is a deliberate, reportable addition."""
    e = env_on()
    hi = B.VolProportionalWidth(3, base_factor=1e4)(None, e)
    lo = B.VolProportionalWidth(3, base_factor=100.0)(None, e)
    assert lo == HOLD
    assert hi != HOLD


def test_the_grid_does_not_silently_collapse_to_one_policy():
    """A ten-configuration grid that is one policy is not a grid, and selecting over it
    is not selection. With the 0 dropped AND base_factor overshooting, every k saturated
    at the widest band, and VolProportionalWidth(k=3) and (k=15) became the same policy
    with the same reward and one distinct action.

    Exercised on a SPREAD grid, because the mechanism cannot be seen on the paper's own
    `{45, 50, 55}`: three widths within 10 of each other, so anything overshooting
    saturates and anything small maps to 0. On that grid the strategy is effectively a
    binary "off, or about 5%", which is worth knowing in itself.
    """
    e = env_on(widths=(10, 20, 30))
    bf = 10.0 / (3 * e.ew_sigma[e.i])            # puts k=3 on the narrowest width
    acts = {k: B.VolProportionalWidth(k, base_factor=bf)(None, e) for k in (3, 5, 9)}
    assert len(set(acts.values())) == 3, f"different k must give different actions: {acts}"


def test_competitors_read_the_papers_sigma():
    """The originals threshold and size off `ew_sigma`, an EWM std of LOG returns with
    alpha=0.05. `env.vol` is a 24h rolling std of SIMPLE returns and belongs to the
    rebuilt state; they are different numbers and swapping them silently changes what
    every competitor does."""
    e = env_on()
    # .to_numpy() on a pandas column can hand back a read-only view; copy to probe.
    e.ew_sigma = np.zeros_like(e.ew_sigma)
    e.vol = np.ones_like(e.vol)          # if a competitor reads vol, it will act
    assert B.VolProportionalWidth(15)(None, e) == HOLD
    assert B.ILMinimizer(168)(None, e) == HOLD
    assert B.ReactiveRecentering(45, 0.01, 999.0)(None, e) != HOLD  # first call deposits
    p = B.ReactiveRecentering(45, 0.01, 999.0)
    p(None, e)                           # deposit
    assert p(None, e) == HOLD, "thresholded on env.vol instead of ew_sigma"


# ------------------------------------------------------------- basic contracts

def test_passive_never_acts():
    e = env_on()
    assert all(B.Passive()(None, e) == HOLD for _ in range(20))


def test_passive_width_sweep_deposits_once_then_holds():
    e = env_on()
    p = B.PassiveWidthSweep(50)
    assert p(None, e) != HOLD
    e.step(p.name and 1)
    assert all(p(None, e) == HOLD for _ in range(10))


def test_recentre_when_out_acts_only_when_out_of_range():
    e = env_on()
    p = B.RecentreWhenOut(50)
    e.step(e._a_enter0)                   # in range now
    assert p(None, e) == HOLD
    e.sqrtA, e.sqrtB = 1e9, 2e9           # force out of range
    assert p(None, e) != HOLD


@pytest.mark.parametrize("cls,kw", [(B.VolProportionalWidth, dict(k=5)),
                                    (B.ILMinimizer, dict(horizon=24))])
def test_only_when_out_holds_while_in_range(cls, kw):
    e = env_on()
    e.step(e._a_enter0)
    p = cls(only_when_out=True, **kw)
    assert p(None, e) == HOLD
