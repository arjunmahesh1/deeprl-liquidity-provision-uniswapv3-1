"""Does removing the martingale from the reward make the problem learnable?

The claim under test. The LP reward is fee - dIL, and one step of dIL against the
hold basket is

    dIL = (hold0 - a0) * dP  -  (1/2) * a0'(P) * dP^2

The first term is first-order in the price move, is a martingale, and no action
influences it. The second is the only part a policy steers. The noise term is much
larger, so the policy gradient is mostly noise. Benchmarking against a portfolio that
holds the position's CURRENT amounts for one step (loss-versus-rebalancing) cancels
the first-order term exactly, because that benchmark's exposure IS a0*dP.

So: train PPO on each reward, evaluate every arm on the SAME unshaped true reward.
Shaping is a control variate, never a change to the reported metric.

    none    fee - dIL                    the true reward, as reported
    shadow  minus a passive shadow       hand-built control variate
    lvr     fee - LVR - costs            the literature's decomposition

If `lvr` does not beat `none` on the true reward, the reformulation is not worth a
paper and the variance argument does not survive contact with the data.

Selection is on VALIDATION only. This script never reads test: it answers whether the
shaping helps, not what the final number is.

Usage:  python scripts/exhibits/exhibit_lvr_shaping.py [--pools ...] [--steps 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.deeprl_liquidity_provision_uniswapv3.agents.registry import make_agent  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.experiments.agent_arm import (  # noqa: E402
    SEEDS, score_agent, train_env,
)
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import (  # noqa: E402
    WINDOW, load_panel, make_split,
)

SHAPINGS = ["none", "shadow", "lvr"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=CORE)
    ap.add_argument("--widths", nargs="*", type=float, default=[100, 200, 500])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()

    splits = {k: make_split(len(load_panel(k)) // WINDOW) for k in args.pools}

    print(f"{args.algo.upper()} ({args.schedule}), {args.steps:,} steps, "
          f"{len(SEEDS)} seeds. Trained on each reward, ALL scored on the true one.")
    print("Validation only; test is not read here.\n")

    out = {}
    for shaping in SHAPINGS:
        per_pool, acts = {}, []
        for k in args.pools:
            vals = []
            for s in SEEDS:
                env = train_env(k, args.widths, splits[k], args.schedule,
                                reward_shaping=shaping)
                m = make_agent(args.algo, env, seed=s, learning_rate=3e-4, ent_coef=0.01)
                m.learn(total_timesteps=args.steps)
                # scored on an UNSHAPED env: score() reads the true reward
                vals.append(score_agent(k, m, args.widths, splits[k].val,
                                        args.schedule).mean())
                from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import (
                    build_env, score,
                )
                from src.deeprl_liquidity_provision_uniswapv3.experiments.agent_arm import (
                    AgentPolicy,
                )
                e = build_env(k, splits[k].val[0], args.widths, schedule=args.schedule)
                acts.append(score(e, AgentPolicy(m, "a"))[1])
            per_pool[k] = float(np.mean(vals))
        out[shaping] = per_pool
        pooled = float(np.mean(list(per_pool.values())))
        # A policy using one action is not controlling anything, whatever it scores.
        print(f"  {shaping:<7} pooled val {pooled:>9,.0f}   "
              f"distinct actions {np.mean(acts):>4.1f}   "
              + "  ".join(f"{k.split('_')[0]}{k[-3:]} {v:>7,.0f}" for k, v in per_pool.items()))

    print(f"\n{'shaping':<10} {'pooled val (true reward)':>26} {'vs none':>10}")
    print("-" * 50)
    base = float(np.mean(list(out["none"].values())))
    for s in SHAPINGS:
        p = float(np.mean(list(out[s].values())))
        print(f"{s:<10} {p:>26,.0f} {p - base:>+10,.0f}")


if __name__ == "__main__":
    main()
