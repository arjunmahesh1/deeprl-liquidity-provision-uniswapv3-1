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
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import wilcoxon

from ..agents.registry import make_agent
from ..data.pools import CORE
from ..policies import baselines as B
from .agent_arm import AgentPolicy, score_agent, train_env
from .bakeoff import (WINDOW, Split, build_env, load_panel, score)

N_TRAIN, N_VAL = 4, 1


def rolling_steps(n_windows: int, n_train: int = N_TRAIN, n_val: int = N_VAL) -> list[Split]:
    """One Split per step. `Split` asserts train/val/test are disjoint."""
    steps, block = [], n_train + n_val
    for i in range(n_windows - block):
        steps.append(Split(train=list(range(i, i + n_train)),
                           val=[i + n_train],
                           test=[i + block]))
    return steps


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
    return dict(test=r, name=best.name, n_actions=n_act)


def agent_step(key, split, widths, algo, schedule, steps, seeds, shaping):
    """Fit on this step's train block, select the seed on validation, test once.

    Refit from scratch every step: carrying weights forward would make the effective
    training set the whole history and quietly undo the walk-forward.
    """
    models, vals = [], []
    for s in seeds:
        env = train_env(key, widths, split, schedule, reward_shaping=shaping)
        m = make_agent(algo, env, seed=s, learning_rate=3e-4, ent_coef=0.01)
        m.learn(total_timesteps=steps)
        models.append(m)
        vals.append(score_agent(key, m, widths, split.val, schedule).mean())
    # Seeds are averaged on TEST, not argmaxed: picking the best seed by test reward
    # is the previous paper's max-over-trials leak wearing a different hat.
    per_seed = [score_agent(key, m, widths, split.test, schedule) for m in models]
    e = build_env(key, split.test[0], widths, schedule=schedule)
    n_act = score(e, AgentPolicy(models[0], "a"))[1] if e is not None else 0
    return dict(test=float(np.mean([p.mean() for p in per_seed])),
                val=float(np.mean(vals)), n_actions=n_act)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=CORE)
    ap.add_argument("--widths", nargs="*", type=float, default=[100, 200, 500])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    ap.add_argument("--shaping", default="none", choices=["none", "shadow", "lvr"])
    ap.add_argument("--max-steps", type=int, default=None, help="cap rolling steps (debug)")
    args = ap.parse_args()

    print(__doc__.split("\n\n")[0])
    print(f"\ntrain {N_TRAIN} windows -> val 1 -> test 1, rolling by 1. "
          f"{args.algo.upper()} ({args.schedule}), {args.steps:,} steps, "
          f"seeds {args.seeds}, shaping {args.shaping}\n")

    arms = {"Passive": lambda w: [B.Passive()],
            "RecentreWhenOut": lambda w: [B.RecentreWhenOut(x) for x in w],
            "ILMinimizer": lambda w: [B.ILMinimizer(h, only_when_out=o)
                                      for h in (24, 168) for o in (False, True)],
            "VolProportionalWidth": lambda w: [B.VolProportionalWidth(k, only_when_out=o)
                                               for k in (3, 5, 7, 10, 15) for o in (False, True)],
            "ReactiveRecentering": lambda w: [B.ReactiveRecentering(x, v, j) for x in w
                                              for v in (0.005, 0.01, 0.02)
                                              for j in (0.005, 0.01, 0.02)],
            "PassiveWidthSweep": lambda w: [B.PassiveWidthSweep(x) for x in w]}

    res = {a: [] for a in arms}
    res[args.algo.upper()] = []
    acts = {a: [] for a in res}

    for key in args.pools:
        steps = rolling_steps(len(load_panel(key)) // WINDOW)
        if args.max_steps:
            steps = steps[:args.max_steps]
        print(f"{key}: {len(steps)} rolling steps")
        for split in steps:
            for name, build in arms.items():
                r = heuristic_step(key, split, args.widths, build(args.widths))
                if r:
                    res[name].append(r["test"])
                    acts[name].append(r["n_actions"])
            r = agent_step(key, split, args.widths, args.algo, args.schedule,
                           args.steps, args.seeds, args.shaping)
            res[args.algo.upper()].append(r["test"])
            acts[args.algo.upper()].append(r["n_actions"])

    n = min(len(v) for v in res.values())
    for k in res:
        res[k] = np.asarray(res[k][:n])
    order = sorted(res, key=lambda k: -res[k].mean())
    passive = res["Passive"].mean()
    champ = order[0]

    print(f"\nWALK-FORWARD TEST, every window tested once by a policy fit only on the "
          f"{N_TRAIN} windows before it (n={n} window-tests per arm)")
    print(f"{'strategy':<24} {'TEST':>9} {'mitigation':>11} {'vs best':>9} {'actions':>8}")
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


if __name__ == "__main__":
    main()
