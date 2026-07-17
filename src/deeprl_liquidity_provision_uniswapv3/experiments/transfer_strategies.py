"""H4: does a strategy selected on one pool work on another?

Weight transfer is the wrong question here. H2 found every RL algorithm converges to
roughly passive, so transferring their weights would show every source "transferring
perfectly" to every target while measuring nothing. That is precisely how the
previous paper reported 88.5% from two columns whose values were bit-identical
across different source pools.

The question that survives H2 is strategy portability: select a competitor on pool
A's VALIDATION, then score it on pool B's TEST. That is what a practitioner faces,
and H3 makes it falsifiable. H3 found the asset pair dominates mitigation while the
fee tier barely moves it, so a strategy should port within pair, across tier, and
should fail across pairs.

The 3x2 panel is what separates the two:
    within pair, across tier   USDC/WETH 0.05% -> USDC/WETH 0.30%   asset fixed
    across pair, within tier   USDC/WETH 0.05% -> WBTC/WETH 0.05%   tier fixed
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from ..data.pools import POOLS
from ..policies import baselines as B
from .bakeoff import PANEL, WINDOW, evaluate_on_test, make_split, select_on_val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", nargs="*", type=float, default=[100, 200, 500, 2000])
    args = ap.parse_args()

    core = [k for k, p in POOLS.items() if p.group == "core"]
    splits = {k: make_split(len(pd.read_parquet(PANEL / f"{k}_hourly.parquet")) // WINDOW)
              for k in core}

    # One selection per pool, on that pool's validation. Test is never consulted.
    print("Selecting one competitor per pool on its own validation ...")
    chosen = {}
    for k in core:
        pol, val = select_on_val(k, B.candidate_grid(args.widths), args.widths, splits[k])
        chosen[k] = pol
        print(f"  {k:<16} {pol.name}")

    # Score every source's choice on every target's test.
    grid, passive = {}, {}
    for tgt in core:
        passive[tgt] = evaluate_on_test(tgt, B.Passive(), args.widths, splits[tgt]).mean()
    for src, tgt in product(core, core):
        r = evaluate_on_test(tgt, chosen[src], args.widths, splits[tgt])
        grid[(src, tgt)] = float(r.mean())

    short = lambda k: k.replace("_weth", "/W").replace("weth_", "W/")
    print(f"\nMitigation over passive on the TARGET's test windows (rows: source of the strategy)")
    print(f"{'':<16}" + "".join(f"{short(t):>12}" for t in core))
    for src in core:
        print(f"{src:<16}" + "".join(f"{grid[(src,t)]-passive[t]:>12,.0f}" for t in core))

    # Does the source matter at all? If not, the whole exercise is vacuous.
    print("\nSpread across sources per target (0 => the source is irrelevant, H4 void):")
    for tgt in core:
        col = np.array([grid[(s, tgt)] for s in core])
        print(f"  {tgt:<16} sd {col.std():>9,.0f}")

    # The two contrasts.
    print(f"\n{'contrast':<26} {'transfer':<22} {'retained':>9}")
    print("-" * 60)
    rows = {"within pair, across tier": [], "across pair, within tier": []}
    for src, tgt in product(core, core):
        if src == tgt:
            continue
        ps, ts = POOLS[src].pair, POOLS[src].fee_tier
        pt, tt = POOLS[tgt].pair, POOLS[tgt].fee_tier
        if ps == pt:
            kind = "within pair, across tier"
        elif ts == tt:
            kind = "across pair, within tier"
        else:
            continue
        native = grid[(tgt, tgt)] - passive[tgt]
        moved = grid[(src, tgt)] - passive[tgt]
        frac = moved / native if native != 0 else np.nan
        rows[kind].append(frac)
        print(f"{kind:<26} {src[:10]+'->'+tgt[:10]:<22} {frac:>9.2f}")
    print()
    for kind, v in rows.items():
        v = np.array([x for x in v if np.isfinite(x)])
        print(f"{kind:<26} median retained {np.median(v):>6.2f}   (n={len(v)})")
    print("\nH3 predicts: within pair ~1 (tier does not matter), across pair << 1.")


if __name__ == "__main__":
    main()
