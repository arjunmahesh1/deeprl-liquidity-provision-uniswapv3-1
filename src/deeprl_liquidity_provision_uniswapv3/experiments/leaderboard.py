"""The full field: every strategy family against every RL algorithm.

Each FAMILY (the paper's four competitors, plus Passive and RecentreWhenOut) gets
its best configuration selected on pooled validation, then exactly one test read.
Families are a pre-specified set, not a search, and the family-wise error is
controlled with Holm across the comparisons against the winner.

Reporting every candidate's test score would be a multiple test read, i.e. the leak
in another costume, so candidate-level results are shown on VALIDATION only.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..agents.registry import ALGOS, make_agent
from ..data.pools import POOLS
from ..policies import baselines as B
from .agent_arm import PPO_GRID, SEEDS, score_agent, train_env
from .bakeoff import PANEL, WINDOW, build_env, make_split, score

FAMILIES = {
    "Passive (never act)": lambda w: [B.Passive()],
    "PassiveWidthSweep": lambda w: [B.PassiveWidthSweep(x) for x in w],
    "VolProportionalWidth": lambda w: [B.VolProportionalWidth(k, only_when_out=o)
                                       for k in (3, 5, 7, 10, 15) for o in (False, True)],
    "ILMinimizer": lambda w: [B.ILMinimizer(h, only_when_out=o)
                              for h in (24, 168) for o in (False, True)],
    "ReactiveRecentering": lambda w: [B.ReactiveRecentering(x, v, j) for x in w
                                      for v in (0.005, 0.01, 0.02)
                                      for j in (0.005, 0.01, 0.02)],
    "RecentreWhenOut": lambda w: [B.RecentreWhenOut(x) for x in w],
}


def pooled(keys, splits, pol, widths, which):
    rs = []
    for k in keys:
        for wi in getattr(splits[k], which):
            e = build_env(k, wi, widths)
            if e is not None:
                rs.append(score(e, pol)[0])
    return np.asarray(rs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()

    core = [k for k, p in POOLS.items() if p.group == "core"]
    splits = {k: make_split(len(pd.read_parquet(PANEL / f"{k}_hourly.parquet")) // WINDOW)
              for k in core}

    rows = []
    print("Selecting each family's best config on pooled validation ...")
    for fam, build in FAMILIES.items():
        cands = build(args.widths)
        best, bv = None, -np.inf
        for c in cands:
            v = pooled(core, splits, c, args.widths, "val").mean()
            if v > bv:
                best, bv = c, v
        te = np.concatenate([pooled([k], splits, best, args.widths, "test") for k in core])
        rows.append(dict(name=fam, cfg=best.name, val=bv, test=te, kind="heuristic"))
        print(f"  {fam:<24} {best.name:<42} val {bv:>8,.0f}")

    print(f"\nTraining each RL algorithm ({args.schedule}); one config on pooled validation ...")
    for algo in ("ppo", "a2c", "dqn"):
        best_i, best_v, store = None, -np.inf, {}
        for i, cfg in enumerate(PPO_GRID):
            store[i] = {k: [make_agent(algo, train_env(k, args.widths, splits[k], args.schedule),
                                       seed=s, **cfg) for s in SEEDS] for k in core}
            for k in core:
                for m in store[i][k]:
                    m.learn(total_timesteps=args.steps)
            v = np.mean([score_agent(k, m, args.widths, splits[k].val, args.schedule).mean()
                         for k in core for m in store[i][k]])
            if v > best_v:
                best_i, best_v = i, v
        te = np.concatenate([np.vstack([score_agent(k, m, args.widths, splits[k].test,
                                                    args.schedule)
                                        for m in store[best_i][k]]).mean(axis=0) for k in core])
        rows.append(dict(name=f"{algo.upper()} ({args.schedule})", cfg=str(PPO_GRID[best_i]),
                         val=best_v, test=te, kind="rl"))
        print(f"  {algo.upper():<24} {str(PPO_GRID[best_i]):<42} val {best_v:>8,.0f}")

    rows.sort(key=lambda r: -r["test"].mean())
    # exact match: "Passive (never act)". startswith("Passive") also catches
    # PassiveWidthSweep, which silently made mitigation relative to the wrong arm.
    passive = next(r for r in rows if r["name"] == "Passive (never act)")["test"].mean()
    champ = rows[0]

    print(f"\n{'strategy':<26} {'kind':<10} {'val':>8} {'TEST':>9} {'mitigation':>11} {'vs best':>9}")
    print("-" * 78)
    for r in rows:
        m = r["test"].mean()
        print(f"{r['name']:<26} {r['kind']:<10} {r['val']:>8,.0f} {m:>9,.0f} "
              f"{m-passive:>+11,.0f} {m-champ['test'].mean():>+9,.0f}")

    # Holm across the family, paired by window, against the winner.
    print(f"\nPaired against {champ['name']}, by window (n={len(champ['test'])}), Holm-corrected:")
    tests = []
    for r in rows[1:]:
        d = r["test"] - champ["test"]
        _, p = wilcoxon(d)
        tests.append((r["name"], d.mean(), (d > 0).mean(), p))
    tests.sort(key=lambda t: t[3])
    m = len(tests)
    print(f"{'strategy':<26} {'diff':>9} {'wins':>6} {'p raw':>9} {'p Holm':>9} {'sig':>5}")
    for i, (nm, dm, w, p) in enumerate(tests):
        ph = min(1.0, p * (m - i))
        print(f"{nm:<26} {dm:>+9,.0f} {w:>5.0%} {p:>9.4f} {ph:>9.4f} {'*' if ph < 0.05 else '':>5}")


if __name__ == "__main__":
    main()
