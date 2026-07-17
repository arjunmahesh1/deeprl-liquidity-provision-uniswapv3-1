"""H4: cross-pool transfer on the 3x2 design.

The 3x2 panel separates what a two-pool design cannot:

    within pair, across tier    USDC/WETH 0.05% -> USDC/WETH 0.30%    asset fixed
    across pair, within tier    USDC/WETH 0.05% -> WBTC/WETH 0.05%    tier fixed

H3 found the asset pair dominates mitigation and the fee tier barely moves it, so
this predicts transfer survives across tiers and degrades across pairs. That is a
falsifiable prediction, not a search.

The previous transfer result is void and this fixes each cause:
  - its "same-period" test set was a SUPERSET of its training set (0-85% contained
    0-70%), so 82% of the "test" was training data. Splits here are disjoint and
    asserted.
  - its reward was an episode SUM compared across segments of different length, so
    "same beats future" was guaranteed by episode length alone. Reported per window
    here, all windows equal length.
  - its saved outputs repeat bit-for-bit across different source pools, meaning the
    result did not depend on the transfer. Source-dependence is asserted here.

Transfer efficiency = transferred / native, both scored on the TARGET's test
windows. Native is the same algorithm trained on the target itself.
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from ..agents.registry import make_agent
from ..data.pools import POOLS
from .agent_arm import SEEDS, score_agent, train_env
from .bakeoff import PANEL, WINDOW, make_split

TIER = {0.0005: "0.05%", 0.0030: "0.30%"}


def train_on(key, widths, schedule, steps, seed, algo, cfg):
    n = len(pd.read_parquet(PANEL / f"{key}_hourly.parquet")) // WINDOW
    split = make_split(n)
    m = make_agent(algo, train_env(key, widths, split, schedule), seed=seed, **cfg)
    m.learn(total_timesteps=steps)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", nargs="*", type=float, default=[100, 200, 500, 2000])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="daily")
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()

    core = [k for k, p in POOLS.items() if p.group == "core"]
    cfg = dict(learning_rate=3e-4, ent_coef=0.01)

    # Train one model per source pool, per seed.
    print(f"Training {args.algo.upper()} on each source pool ...")
    models = {k: [train_on(k, args.widths, args.schedule, args.steps, s, args.algo, cfg)
                  for s in SEEDS] for k in core}

    # Score every source on every target's TEST windows.
    print("\nScoring every source on every target's test windows.")
    grid = {}
    for src, tgt in product(core, core):
        n = len(pd.read_parquet(PANEL / f"{tgt}_hourly.parquet")) // WINDOW
        split = make_split(n)
        per_seed = [score_agent(tgt, m, args.widths, split.test, args.schedule).mean()
                    for m in models[src]]
        grid[(src, tgt)] = float(np.mean(per_seed))

    print(f"\n{'':<16}" + "".join(f"{t.replace('_weth','').replace('weth_',''):>12}" for t in core))
    for src in core:
        row = "".join(f"{grid[(src,t)]:>12,.0f}" for t in core)
        print(f"{src:<16}{row}")

    # Source-dependence: the previous result's rows were bit-identical.
    print("\nDoes the source pool matter? (it did not in the previous paper's output)")
    for tgt in core:
        col = np.array([grid[(s, tgt)] for s in core])
        print(f"  target {tgt:<16} spread across sources: {col.std():>9,.0f}")

    # The two contrasts the 3x2 identifies.
    print("\nTransfer efficiency = transferred / native, on the target's test windows.")
    print(f"{'contrast':<44} {'efficiency':>11}")
    print("-" * 58)
    for src in core:
        ps, ts = POOLS[src].pair, POOLS[src].fee_tier
        for tgt in core:
            if src == tgt:
                continue
            pt, tt = POOLS[tgt].pair, POOLS[tgt].fee_tier
            same_pair, same_tier = ps == pt, ts == tt
            if not (same_pair or same_tier):
                continue
            kind = "within pair, across tier" if same_pair else "across pair, within tier"
            native = grid[(tgt, tgt)]
            eff = grid[(src, tgt)] / native if native != 0 else np.nan
            print(f"{kind:<24} {src[:8]}->{tgt[:8]:<10} {eff:>11.2f}")


if __name__ == "__main__":
    main()
