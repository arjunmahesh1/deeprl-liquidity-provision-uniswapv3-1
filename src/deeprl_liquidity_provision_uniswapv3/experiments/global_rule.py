"""The decisive comparison: one globally-selected rule against the agent.

Per-pool selection is the confound. H4 showed it fails on the WBTC pools:
`wbtc_weth_005`'s own validation picks `PassiveWidthSweep(w=2000)` worth +366 on
its test, while `ILMinimizer(H=24,out)` imported from another pool scores +1,458 on
that same test. So "PPO beats the best competitor" on those pools measures a bad
selection, not a good agent; and on the other pools per-pool selection flatters the
competitor instead. Either way the per-pool baseline is noise.

This removes it. One rule is selected on VALIDATION POOLED ACROSS ALL POOLS, then
scored once on test. The agent gets the identical treatment: one config, chosen on
pooled validation, scored once on test. Neither side gets to pick per pool.

Reads test exactly once per arm, after selection.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..agents.registry import make_agent
from ..data.pools import POOLS
from ..policies import baselines as B
from .agent_arm import PPO_GRID, SEEDS, score_agent, train_env
from .bakeoff import PANEL, WINDOW, build_env, make_split, score


def pooled_val_score(keys, splits, pol, widths):
    """Mean reward across every pool's validation windows. Test is untouched."""
    rs = []
    for k in keys:
        for wi in splits[k].val:
            env = build_env(k, wi, widths)
            if env is not None:
                rs.append(score(env, pol)[0])
    return float(np.mean(rs)) if rs else -np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()

    core = [k for k, p in POOLS.items() if p.group == "core"]
    splits = {k: make_split(len(pd.read_parquet(PANEL / f"{k}_hourly.parquet")) // WINDOW)
              for k in core}

    # --- one rule for the whole panel, chosen on pooled validation -----------
    print("Selecting ONE rule on validation pooled across all pools ...")
    cands = B.candidate_grid(args.widths)
    scored = [(pooled_val_score(core, splits, p, args.widths), p) for p in cands]
    scored.sort(key=lambda t: -t[0])
    best_val, best_rule = scored[0]
    print(f"  chosen: {best_rule.name}   (pooled val {best_val:,.0f})")
    print("  runners-up: " + ", ".join(f"{p.name} ({v:,.0f})" for v, p in scored[1:4]))

    # --- one agent config for the whole panel, chosen on pooled validation ---
    print(f"\nTraining {args.algo.upper()} ({args.schedule}) per pool; "
          f"ONE config chosen on pooled validation ...")
    models = {}
    for cfg_i, cfg in enumerate(PPO_GRID):
        models[cfg_i] = {k: [] for k in core}
        for k in core:
            for s in SEEDS:
                m = make_agent(args.algo, train_env(k, args.widths, splits[k], args.schedule),
                               seed=s, **cfg)
                m.learn(total_timesteps=args.steps)
                models[cfg_i][k].append(m)
    best_cfg_i, best_cfg_val = None, -np.inf
    for cfg_i, cfg in enumerate(PPO_GRID):
        v = np.mean([score_agent(k, m, args.widths, splits[k].val, args.schedule).mean()
                     for k in core for m in models[cfg_i][k]])
        if v > best_cfg_val:
            best_cfg_i, best_cfg_val = cfg_i, v
    print(f"  chosen: {PPO_GRID[best_cfg_i]}   (pooled val {best_cfg_val:,.0f})")

    # --- the single test read ------------------------------------------------
    print(f"\n{'pool':<16} {'passive':>9} {'global rule':>12} {'agent':>9} "
          f"{'rule-pass':>10} {'agent-pass':>11} {'agent-rule':>11}")
    print("-" * 82)
    R, A, P = [], [], []
    for k in core:
        te = splits[k].test
        p_ = np.array([score(build_env(k, wi, args.widths), B.Passive())[0] for wi in te])
        r_ = np.array([score(build_env(k, wi, args.widths), best_rule)[0] for wi in te])
        a_ = np.vstack([score_agent(k, m, args.widths, te, args.schedule)
                        for m in models[best_cfg_i][k]]).mean(axis=0)
        R.append(r_); A.append(a_); P.append(p_)
        print(f"{k:<16} {p_.mean():>9,.0f} {r_.mean():>12,.0f} {a_.mean():>9,.0f} "
              f"{r_.mean()-p_.mean():>+10,.0f} {a_.mean()-p_.mean():>+11,.0f} "
              f"{a_.mean()-r_.mean():>+11,.0f}")
    R, A, P = np.concatenate(R), np.concatenate(A), np.concatenate(P)
    print("-" * 82)
    print(f"{'POOLED':<16} {P.mean():>9,.0f} {R.mean():>12,.0f} {A.mean():>9,.0f} "
          f"{R.mean()-P.mean():>+10,.0f} {A.mean()-P.mean():>+11,.0f} "
          f"{A.mean()-R.mean():>+11,.0f}")

    # Paired at the window level: the independent unit.
    from scipy.stats import wilcoxon
    d = A - R
    stat, p = wilcoxon(d)
    print(f"\nagent - rule, paired by window: mean {d.mean():+,.0f}, "
          f"median {np.median(d):+,.0f}, wins {(d>0).mean():.0%} of {len(d)} windows")
    print(f"Wilcoxon signed-rank: p = {p:.4f}")


if __name__ == "__main__":
    main()
