"""Walk-forward rolling windows: the previous paper's protocol, with the leak removed.

The previous paper trained on five consecutive 1,500-hour windows and tested on the
next one, rolling forward one window at a time and retraining at every step:

    for i in range(len(dfs_list) - 5):
        uni_train = concat(dfs_list[i : i+5])
        uni_test  = dfs_list[i+5]

That design was sound and is what a liquidity provider actually runs: refit on recent
history, deploy on what comes next. The defect was never the rolling window. It was
that `optimize_ppo` received `uni_test` and returned the reward ON IT as the Optuna
objective, so the tuning maximized test reward, INSIDE each rolling step. The single
chronological 50/25/25 split that briefly replaced this removed the leak by removing
the protocol, which also opened a multi-year gap between fitting and testing that the
original never had, and then made the resulting failure to generalize look like a
property of the problem.

The fix keeps the rolling window and carves validation out of the TRAINING block:

    train  windows i .. i+3     6,000h    fit here
    val    window  i+4          1,500h    select here; test is not a parameter
    test   window  i+5          1,500h    read once, then roll

Each window is therefore tested exactly once, by a policy fit only on data preceding
it, with the gap between fitting and testing held to one window. Both arms, learned
and heuristic, get the identical treatment at every step.

## Running it

One WORK UNIT is one (pool, rolling step): 144 of them on the core panel, and no unit
depends on any other. A unit writes a self-describing JSON result and is skipped if
that file already exists, so a run is resumable and a laptop and the cluster can
write into the same directory.

    # everything, here
    python -m ...experiments.rolling --out outputs/rolling_v1

    # one shard of eight (the laptop path; run the other shards elsewhere)
    python -m ...experiments.rolling --out outputs/rolling_v1 --shard 0 --of 8

    # a SLURM array maps $SLURM_ARRAY_TASK_ID onto the SAME index
    sbatch scripts/slurm/rolling_array.sh          # see that file

    # read whatever has landed and print the table
    python -m ...experiments.rolling --out outputs/rolling_v1 --aggregate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from ..agents.registry import make_agent
from ..data.pools import CORE
from ..policies import baselines as B
from .agent_arm import AgentPolicy, score_agent, train_env
from .bakeoff import WINDOW, Split, build_env, load_panel, score

N_TRAIN, N_VAL = 4, 1

# The agent's search budget, selected on validation like every heuristic arm.
#
# Sized against what the heuristics actually search under `--widths 100 200 500`:
# Passive 1, PassiveWidthSweep 3, RecentreWhenOut 3, ILMinimizer 4,
# VolProportionalWidth 10, ReactiveRecentering 27. Eight puts the agent inside that
# range rather than at either extreme. It is NOT free: every config is fit per seed
# per rolling step per pool, so 8 x 2 x 24 x 6 = 2,304 fits before the refits.
#
# `net_arch` spans the previous paper's Optuna-selected [4, 2] against a
# conventional [64, 64], because the two differ by ~3 orders of magnitude in
# capacity and the paper's choice is not obviously right for this state.
AGENT_GRID = [
    dict(learning_rate=lr, ent_coef=ec, net_arch=na)
    for lr in (3e-4, 1e-3)
    for ec in (0.0, 0.01)
    for na in ([4, 2], [64, 64])
]

HEURISTICS = {
    "Passive": lambda w: [B.Passive()],
    "RecentreWhenOut": lambda w: [B.RecentreWhenOut(x) for x in w],
    "ILMinimizer": lambda w: [B.ILMinimizer(h, only_when_out=o)
                              for h in (24, 168) for o in (False, True)],
    "VolProportionalWidth": lambda w: [B.VolProportionalWidth(k, only_when_out=o)
                                       for k in (3, 5, 7, 10, 15) for o in (False, True)],
    "ReactiveRecentering": lambda w: [B.ReactiveRecentering(x, v, j) for x in w
                                      for v in (0.005, 0.01, 0.02)
                                      for j in (0.005, 0.01, 0.02)],
    "PassiveWidthSweep": lambda w: [B.PassiveWidthSweep(x) for x in w],
}


def rolling_steps(n_windows: int, n_train: int = N_TRAIN, n_val: int = N_VAL) -> list[Split]:
    """One Split per step. `Split` asserts train/val/test are disjoint."""
    steps, block = [], n_train + n_val
    for i in range(n_windows - block):
        steps.append(Split(train=list(range(i, i + n_train)),
                           val=[i + n_train],
                           test=[i + block]))
    return steps


def config_tag(args) -> str:
    """A short stable hash of everything that changes what a unit COMPUTES.

    The resume key used to be (pool, step) alone. Because a unit whose file exists is
    skipped, running `--algo a2c` into a directory that already held a PPO run
    silently skipped every unit and then aggregated the PPO results under the A2C
    name. Changing `--widths`, `--schedule`, `--shaping` or `--steps` did the same,
    only worse: the directory ends up holding a mix that `aggregate` pools into one
    table without noticing. The config now keys the filename, so a changed config is
    a different unit rather than a silent no-op.
    """
    payload = json.dumps({k: v for k, v in sorted(vars(args).items())
                          if k in ("algo", "schedule", "widths", "shaping", "steps",
                                   "seeds", "paper_extractor")}, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Unit:
    """One (pool, rolling step). The atom of work, identical on any machine."""
    pool: str
    step: int
    tag: str = "untagged"

    @property
    def name(self) -> str:
        return f"{self.pool}__step{self.step:03d}__{self.tag}"


def work_units(pools: list[str], tag: str = "untagged") -> list[Unit]:
    """Every unit, in a deterministic order, so `--shard i --of n` means the same
    thing on the laptop and on the cluster."""
    out = []
    for key in sorted(pools):
        for s in range(len(rolling_steps(len(load_panel(key)) // WINDOW))):
            out.append(Unit(key, s, tag))
    return out


def provenance(args) -> dict:
    """Enough to tell two runs apart, and to tell a laptop shard from a cluster one."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", True
    return {"git_sha": sha, "git_dirty": dirty, "host": platform.node(),
            "platform": platform.platform(), "python": platform.python_version(),
            "config": {k: v for k, v in vars(args).items()
                       if k not in ("shard", "of", "aggregate", "out")}}


# --------------------------------------------------------------- the two arms

def heuristic_step(key, split, widths, candidates):
    """Select on this step's validation window, score once on its test window."""
    best, best_v = None, -np.inf
    for pol in candidates:
        e = build_env(key, split.val[0], widths)
        if e is None:
            continue
        v = score(e, pol)[0]
        if v > best_v:
            best, best_v = pol, v
    e = build_env(key, split.test[0], widths)
    if e is None or best is None:
        return None
    r, n_act = score(e, best)
    return dict(test=r, val=best_v, name=best.name, n_actions=n_act)


def agent_step(key, split, widths, algo, schedule, steps, seeds, shaping,
               paper_extractor=False):
    """Select a config on validation, refit on train+val, score once on test.

    Two defects this replaces, both of which rigged the comparison AGAINST the agent
    while `bakeoff` and `agent_arm` asserted the budgets were matched:

    1. The config was hardcoded (`learning_rate=3e-4, ent_coef=0.01`) and the
       validation score was computed and then DISCARDED. The agent searched exactly
       ONE configuration while `ReactiveRecentering` searched 27 and the heuristics
       searched 48 between them. The previous paper had the opposite bias (10 Optuna
       trials for PPO against a competitor with no free parameters at all); replacing
       one rigged comparison with its mirror image is not a fix.

    2. The agent fit on `train` only, so its data ended one window BEFORE its test
       window, while every heuristic selected on `val`, i.e. right up to the test
       boundary. Under this project's own diagnosis, that a regime shift between
       selection and test is what breaks a learned policy, that asymmetry pointed
       straight at the result. Refitting on train+val after selection costs nothing,
       restores the previous paper's 7,500h fitting block, and puts both arms'
       information on the same footing.

    Refit from scratch every step: carrying weights forward would make the effective
    training set the whole history and quietly undo the walk-forward.
    """
    # --- select on validation. Test is not reachable from this loop. -----------
    best_cfg, best_v = None, -np.inf
    for cfg in AGENT_GRID:
        vals = []
        for s in seeds:
            env = train_env(key, widths, split, schedule, reward_shaping=shaping)
            m = make_agent(algo, env, seed=s, paper_extractor=paper_extractor, **cfg)
            m.learn(total_timesteps=steps)
            vals.append(float(score_agent(key, m, widths, split.val, schedule).mean()))
        v = float(np.mean(vals))
        if v > best_v:
            best_cfg, best_v = cfg, v

    # --- refit on train+val with the chosen config, then read test ONCE --------
    # val is EMPTY here, not repeated into val: selection is already done, and a Split
    # whose train and val overlap is exactly what `Split.__post_init__` exists to
    # refuse. Reusing the same object for "fit on everything before test" and "select"
    # is how a leak gets in.
    fit = Split(train=split.train + split.val, val=[], test=split.test)
    models = []
    for s in seeds:
        env = train_env(key, widths, fit, schedule, reward_shaping=shaping)
        m = make_agent(algo, env, seed=s, paper_extractor=paper_extractor, **best_cfg)
        m.learn(total_timesteps=steps)
        models.append(m)
    # Seeds are averaged on TEST, not argmaxed: picking the best seed by test reward
    # is the previous paper's max-over-trials leak wearing a different hat.
    per_seed = [score_agent(key, m, widths, split.test, schedule) for m in models]
    e = build_env(key, split.test[0], widths, schedule=schedule)
    n_act = score(e, AgentPolicy(models[0], "a"))[1] if e is not None else 0
    return dict(test=float(np.mean([p.mean() for p in per_seed])),
                val=best_v, name=f"{algo}/{shaping}/{best_cfg}", n_actions=n_act)


def run_unit(unit: Unit, args) -> dict:
    """Every arm on one (pool, step). Self-contained: no other unit is consulted."""
    split = rolling_steps(len(load_panel(unit.pool)) // WINDOW)[unit.step]
    arms = {}
    for name, build in HEURISTICS.items():
        r = heuristic_step(unit.pool, split, args.widths, build(args.widths))
        if r:
            arms[name] = r
    arms[args.algo.upper()] = agent_step(unit.pool, split, args.widths, args.algo,
                                         args.schedule, args.steps, args.seeds,
                                         args.shaping, args.paper_extractor)
    return {"unit": asdict(unit), "split": {"train": split.train, "val": split.val,
                                            "test": split.test},
            "arms": arms, "provenance": provenance(args)}


# ---------------------------------------------------------------- aggregation

def aggregate(out_dir: Path, expected: list[Unit]) -> None:
    files = sorted(out_dir.glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        print(f"no results in {out_dir}")
        return
    done = {r["unit"]["pool"] + str(r["unit"]["step"]) for r in rows}
    missing = [u for u in expected if u.pool + str(u.step) not in done]

    res, acts = {}, {}
    for r in rows:
        for arm, d in r["arms"].items():
            res.setdefault(arm, []).append(d["test"])
            acts.setdefault(arm, []).append(d["n_actions"])
    res = {k: np.asarray(v) for k, v in res.items()}
    n = len(rows)

    print(f"\nWALK-FORWARD TEST. Every window tested once by a policy fit only on the "
          f"{N_TRAIN} windows before it.")
    print(f"{n} of {len(expected)} work units complete"
          + (f"; MISSING {len(missing)}: {[u.name for u in missing[:6]]}" if missing else ""))
    if missing:
        # A silent partial aggregate reads as a finished run. Say what is absent.
        print("  ^ the table below is a PARTIAL result and is not comparable to a full run")

    order = sorted(res, key=lambda k: -res[k].mean())
    passive = res["Passive"].mean()
    champ = order[0]
    print(f"\n{'strategy':<24} {'TEST':>9} {'mitigation':>11} {'vs best':>9} {'actions':>8}")
    print("-" * 66)
    for k in order:
        print(f"{k:<24} {res[k].mean():>9,.0f} {res[k].mean()-passive:>+11,.0f} "
              f"{res[k].mean()-res[champ].mean():>+9,.0f} {np.mean(acts[k]):>8.1f}")

    print(f"\nPaired against {champ}, by window (n={n}), Holm-corrected:")
    tests = []
    for k in order[1:]:
        d = res[k] - res[champ]
        if np.allclose(d, 0):
            continue
        tests.append((k, d.mean(), (d > 0).mean(), wilcoxon(d)[1]))
    tests.sort(key=lambda t: t[3])
    m = len(tests)
    print(f"{'strategy':<24} {'diff':>9} {'wins':>6} {'p raw':>9} {'p Holm':>9} {'sig':>5}")
    for i, (nm, dm, w, p) in enumerate(tests):
        ph = min(1.0, p * (m - i))
        print(f"{nm:<24} {dm:>+9,.0f} {w:>5.0%} {p:>9.4f} {ph:>9.4f} {'*' if ph < 0.05 else '':>5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="*", default=CORE)
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    ap.add_argument("--shaping", default="none", choices=["none", "shadow", "lvr"])
    ap.add_argument("--paper-extractor", action="store_true",
                    help="use the previous paper's BatchNorm feature extractor")
    ap.add_argument("--out", type=Path, required=True, help="run directory for unit results")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index")
    ap.add_argument("--of", type=int, default=1, help="total shards")
    ap.add_argument("--aggregate", action="store_true", help="read results and report")
    ap.add_argument("--force", action="store_true", help="recompute units already on disk")
    args = ap.parse_args()

    units = work_units(args.pools, config_tag(args))
    if args.aggregate:
        aggregate(args.out, units)
        return

    assert 0 <= args.shard < args.of, f"shard {args.shard} out of range for --of {args.of}"
    args.out.mkdir(parents=True, exist_ok=True)
    mine = units[args.shard::args.of]
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(units)} work units")

    for u in mine:
        path = args.out / f"{u.name}.json"
        if path.exists() and not args.force:
            print(f"  {u.name}  skip (done)")
            continue
        r = run_unit(u, args)
        # Write via a temp file: a shard killed mid-write must not leave a half-parsed
        # result that the aggregate silently counts as complete.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=1))
        tmp.rename(path)
        a = r["arms"]
        print(f"  {u.name}  " + "  ".join(f"{k[:9]} {v['test']:>7,.0f}" for k, v in a.items()))

    if args.shard == 0 and args.of == 1:
        aggregate(args.out, units)


if __name__ == "__main__":
    main()
