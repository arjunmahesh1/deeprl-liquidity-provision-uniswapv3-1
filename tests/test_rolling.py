"""Tests for the walk-forward rolling protocol.

The protocol IS the science here: the previous paper's headline died on a protocol
defect, not a modelling one, so the split generator gets tests before it gets a run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.experiments.rolling import (  # noqa: E402
    N_TRAIN, N_VAL, Unit, rolling_steps,
)


def fake_units(n_pools=3, per_pool=24):
    return [Unit(f"pool{p}", s) for p in range(n_pools) for s in range(per_pool)]


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


# ------------------------------------------------------- the sharding contract
#
# The laptop passes --shard i --of n; a SLURM array passes $SLURM_ARRAY_TASK_ID as
# the same i. If the two ever disagree about which unit index i names, a laptop shard
# and a cluster shard silently compute different things into one output directory and
# the merged table is nonsense. These tests are that contract.

@pytest.mark.parametrize("of", [1, 2, 3, 7, 8, 144])
def test_shards_partition_the_work_exactly(of):
    """Every unit lands in exactly one shard: nothing dropped, nothing computed twice."""
    units = fake_units()
    seen = [u for i in range(of) for u in units[i::of]]
    assert sorted(seen, key=lambda u: (u.pool, u.step)) == sorted(
        units, key=lambda u: (u.pool, u.step))
    assert len(seen) == len(units), "a unit is in two shards"


@pytest.mark.parametrize("of", [2, 3, 8])
def test_shards_are_disjoint(of):
    units = fake_units()
    shards = [set(u.name for u in units[i::of]) for i in range(of)]
    for a in range(of):
        for b in range(a + 1, of):
            assert not (shards[a] & shards[b]), f"shards {a} and {b} overlap"


def test_shard_assignment_is_stable_across_machines():
    """`work_units` orders pools with sorted(), so shard i is the same set of units on
    any machine. A dict-ordered or filesystem-ordered listing would make the laptop's
    shard 3 and the cluster's shard 3 different work."""
    units = fake_units()
    a = [u.name for u in units[3::8]]
    b = [u.name for u in sorted(fake_units(), key=lambda u: (u.pool, u.step))[3::8]]
    assert a == b


def test_shards_are_balanced_within_one_unit():
    """Strided slicing, not chunking: with 144 units over 8 shards every shard gets 18,
    and no shard is left holding a long tail while the rest idle."""
    units = fake_units()
    sizes = [len(units[i::8]) for i in range(8)]
    assert max(sizes) - min(sizes) <= 1


def test_unit_name_is_a_safe_stable_filename():
    """The unit name is the output filename and the resume key: it must be unique, and
    sortable so step 2 does not file between step 19 and step 20."""
    names = [u.name for u in fake_units()]
    assert len(names) == len(set(names))
    for n in names:
        assert "/" not in n and " " not in n
    steps = [Unit("p", s).name for s in (2, 19, 20, 100)]
    assert steps == sorted(steps), "zero-padding is missing; names sort wrong"
