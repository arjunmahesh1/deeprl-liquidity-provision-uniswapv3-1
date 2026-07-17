"""Tests for the walk-forward rolling protocol.

The protocol IS the science here: the previous paper's headline died on a protocol
defect, not a modelling one, so the split generator gets tests before it gets a run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.experiments.rolling import (  # noqa: E402
    N_TRAIN, N_VAL, rolling_steps,
)


def test_every_step_is_disjoint_and_ordered():
    """Train precedes validation precedes test, always, with no overlap. A step that
    validated or tested on data it trained on is the defect that sank the paper."""
    for s in rolling_steps(30):
        assert max(s.train) < s.val[0] < s.test[0]
        assert not (set(s.train) & set(s.val)), "train/val overlap"
        assert not (set(s.train) & set(s.test)), "train/test overlap"
        assert not (set(s.val) & set(s.test)), "val/test overlap"


def test_test_window_immediately_follows_the_fitted_block():
    """The point of walk-forward: the gap between fitting and testing is ONE window.

    The single 50/25/25 split that briefly replaced this opened a gap of years, which
    is not what an LP faces and is not what the previous paper did.
    """
    for s in rolling_steps(30):
        assert s.test[0] == s.val[0] + 1
        assert s.val[0] == max(s.train) + 1


def test_windows_roll_by_one_and_cover_the_panel():
    steps = rolling_steps(30)
    tested = [s.test[0] for s in steps]
    assert tested == list(range(N_TRAIN + N_VAL, 30)), "windows skipped or repeated"
    assert len(steps) == 30 - (N_TRAIN + N_VAL)


def test_each_window_is_tested_exactly_once():
    """One test read per window. Reading a window twice is a multiple test read."""
    tested = [s.test[0] for s in rolling_steps(30)]
    assert len(tested) == len(set(tested))


def test_block_size_matches_the_previous_paper():
    """5 windows of 1,500h fitted, next window tested. The previous paper concatenated
    dfs_list[i:i+5] and tested dfs_list[i+5]; we split that same block into 4 train +
    1 validation so selection never sees test."""
    assert N_TRAIN + N_VAL == 5
    s = rolling_steps(30)[0]
    assert s.train == [0, 1, 2, 3] and s.val == [4] and s.test == [5]


def test_a_panel_too_short_to_roll_yields_no_steps():
    for n in range(0, N_TRAIN + N_VAL + 1):
        assert rolling_steps(n) == []


@pytest.mark.parametrize("n", [12, 20, 29])
def test_step_count_scales_with_the_panel(n):
    assert len(rolling_steps(n)) == max(0, n - (N_TRAIN + N_VAL))
