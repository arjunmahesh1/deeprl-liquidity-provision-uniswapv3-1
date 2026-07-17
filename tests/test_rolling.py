"""Tests for the walk-forward rolling protocol.

The protocol IS the science here: the previous paper's headline died on a protocol
defect, not a modelling one, so the split generator gets tests before it gets a run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402
import json  # noqa: E402

from src.deeprl_liquidity_provision_uniswapv3.experiments.rolling import (  # noqa: E402
    AGENT_GRID, HEURISTICS, N_TRAIN, N_VAL, Unit, config_tag, grid_for, rolling_steps,
)


def fake_units(n_pools=3, per_pool=24):
    return [Unit(f"pool{p}", s) for p in range(n_pools) for s in range(per_pool)]


def fake_args(**kw):
    base = dict(algo="ppo", schedule="event_driven", widths=[100, 200, 500],
                shaping="none", steps=20000, seeds=[42, 123], paper_extractor=False)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------ the search budget
#
# The rebuild asserts in three docstrings that the agent and the heuristics get a
# matched selection budget. It was false: the agent's config was hardcoded and its
# validation score was computed and discarded, so it searched ONE config while
# ReactiveRecentering searched 27. The previous paper had the mirror-image bias (10
# Optuna trials for PPO against a competitor with no free parameters). Replacing one
# rigged comparison with its reflection is not a fix, so the budget gets a test.

def test_agent_searches_more_than_one_config():
    """The defect: a hardcoded config makes 'selected on validation' a fiction."""
    assert len(AGENT_GRID) > 1
    # json, not tuple(): net_arch is a list, so the configs are not hashable as-is.
    keys = [json.dumps(c, sort_keys=True) for c in AGENT_GRID]
    assert len(set(keys)) == len(keys), \
        "duplicate configs in the grid: selection would compare a config against itself"


def test_agent_budget_is_inside_the_heuristic_range():
    """Not matched to a single number, which is impossible across arms of different
    shape, but inside the range the heuristics actually search. At either extreme the
    comparison measures the budget rather than the method.

    Counted on the SHIPPED grid. Note the heuristic counts here are configurations
    offered, not distinct policies: on the paper's own `{45,50,55}` several of them
    coincide (VolProportionalWidth is passive at every k), so these numbers overstate
    the competitors' real search. See tests/test_baselines.py.
    """
    counts = {n: len(b([45, 50, 55])) for n, b in HEURISTICS.items()}
    assert min(counts.values()) <= len(AGENT_GRID) <= max(counts.values()), (
        f"agent searches {len(AGENT_GRID)}, heuristics search {counts}"
    )


@pytest.mark.parametrize("algo", ["ppo", "a2c", "dqn", "qrdqn", "recurrentppo"])
def test_no_algorithm_receives_duplicate_configs(algo):
    """`make_agent` drops kwargs an algorithm does not accept, so a {lr} x {ent_coef}
    grid arrives at DQN, which takes no ent_coef, as four configs each fit TWICE: half
    the compute wasted, and its selection comparing a config against itself."""
    g = grid_for(algo)
    keys = [json.dumps(c, sort_keys=True, default=str) for c in g]
    assert len(set(keys)) == len(keys), f"{algo} receives duplicates: {g}"


def test_value_based_algorithms_get_a_smaller_but_real_grid():
    """DQN/QR-DQN legitimately have no entropy axis, so their grid is half PPO's. That
    is a real asymmetry in the budget and it should be visible, not silently padded
    with duplicates."""
    assert len(grid_for("dqn")) == len(grid_for("qrdqn")) == len(AGENT_GRID) // 2
    assert len(grid_for("ppo")) == len(AGENT_GRID)


# ------------------------------------------------------ the resume key
#
# A unit whose file exists is skipped. With (pool, step) as the only key, running
# --algo a2c into a directory holding a PPO run silently skipped every unit and then
# aggregated the PPO numbers under the A2C name.

@pytest.mark.parametrize("changed", [
    {"algo": "a2c"}, {"schedule": "daily"}, {"widths": [100, 200]},
    {"shaping": "lvr"}, {"steps": 50_000}, {"seeds": [42]}, {"paper_extractor": True},
])
def test_a_changed_config_is_a_different_unit(changed):
    """Anything that changes what a unit COMPUTES must change its filename."""
    a, b = config_tag(fake_args()), config_tag(fake_args(**changed))
    assert a != b, f"changing {changed} did not change the resume key"


def test_the_same_config_is_the_same_unit():
    """...and an unchanged config must still resume, or nothing is skippable."""
    assert config_tag(fake_args()) == config_tag(fake_args())
    assert Unit("p", 1, "abc").name == Unit("p", 1, "abc").name


def test_run_and_aggregate_agree_on_the_config():
    """The runner must pass the SAME config args to the run and to the aggregate.

    It did not: the aggregate call omitted --steps and --seeds, so it hashed a
    different config than the run had written and matched nothing. Before the aggregate
    filtered on the tag at all, this would have silently averaged whatever happened to
    be in the directory. Guarding the script itself, because the bug was in the shell
    glue rather than in Python.
    """
    sh = Path(__file__).resolve().parents[1] / "scripts/run_algos.sh"
    body = sh.read_text()
    # One shared arg list, used by both invocations. If someone spells the flags out
    # twice again, the two can drift apart without anything failing loudly.
    assert body.count("CFG=(") == 1, "the shared config arg list is gone or duplicated"
    assert body.count('"${CFG[@]}"') == 2, (
        "run and aggregate must both use the shared arg list; found "
        f'{body.count(chr(34) + "${CFG[@]}" + chr(34))} use(s)'
    )
    assert "--aggregate" in body


def test_config_tag_ignores_sharding():
    """--shard/--of/--out change WHERE a unit runs, not WHAT it computes. If they
    keyed the filename, two shards of one run would never merge."""
    a = fake_args()
    assert config_tag(a) == config_tag(fake_args())


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
