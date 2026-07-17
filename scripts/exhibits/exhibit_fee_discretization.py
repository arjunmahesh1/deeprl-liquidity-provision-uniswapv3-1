"""What the hourly fee discretization was worth, measured on the real panel.

Claim under test (SPEC.md, "OPEN DEFECT"): crediting a whole hour's fees on an
hour-boundary range check penalises narrow bands, because a narrow band crosses in
and out repeatedly within an hour and any hour ending out of range earns zero. If
that is right, the penalty must grow as the band narrows, and removing it must move
the best passive width toward concentration.

Runs a passive LP at fixed widths over the same windows under both fee models, and
reports fee income and the width each model prefers. Nothing is trained here: this
isolates the accounting.

Usage:  python scripts/exhibits/exhibit_fee_discretization.py [--pool usdc_weth_005]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE, POOLS  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.experiments import bakeoff as BK  # noqa: E402

WIDTHS = [50, 100, 200, 500, 1000, 2000, 5000]


def run(key: str, width: float, wi: int, fee_model: str):
    """One passive LP, one window, one width. Never acts after entry."""
    p = POOLS[key]
    df = BK.load_panel(key)
    seg = df.iloc[wi * BK.WINDOW:(wi + 1) * BK.WINDOW].reset_index(drop=True)
    if len(seg) < BK.WINDOW:
        return None
    env = UniswapV3Env(
        seg, swaps=BK.window_swaps(key, seg) if fee_model == "per_swap" else None,
        fee_model=fee_model, fee_tier_pct=p.fee_tier_pct,
        action_widths=np.array([width]), dec0=p.dec0, dec1=p.dec1,
        capital_usd=30_000.0, gas_usd=5.0, warmup=BK.WARMUP, allow_exit=False,
    )
    env.reset()
    total, fees, il, frac, n = 0.0, 0.0, 0.0, 0.0, 0
    done = trunc = False
    while not (done or trunc):
        _, r, done, trunc, info = env.step(0)      # HOLD forever
        total += r
        fees += info["fee"]
        il += info["d_il"]
        frac += info["range_frac"]
        n += 1
    return dict(reward=total, fees=fees, il=il, in_range=frac / max(n, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=[CORE[0]])
    args = ap.parse_args()

    for key in args.pools:
        n_win = len(BK.load_panel(key)) // BK.WINDOW
        split = BK.make_split(n_win)
        print(f"\n{key}: passive LP held at a fixed width, {len(split.test)} test windows")
        print(f"{'width':>7} {'band':>9} "
              f"{'HOURLY fees':>12} {'reward':>9} {'in-rng':>7}   "
              f"{'PER-SWAP fees':>14} {'reward':>9} {'in-rng':>7}   {'fee lift':>9}")
        print("-" * 104)
        best = {}
        for w in WIDTHS:
            row = {}
            for fm in ("hourly", "per_swap"):
                rs = [run(key, w, wi, fm) for wi in split.test]
                rs = [r for r in rs if r]
                row[fm] = {k: float(np.median([r[k] for r in rs])) for k in rs[0]}
            band = 1.0001 ** w - 1.0
            lift = (row["per_swap"]["fees"] / row["hourly"]["fees"]
                    if row["hourly"]["fees"] > 0 else np.inf)
            print(f"{w:>7.0f} {'+/-'+format(band,'.1%'):>9} "
                  f"{row['hourly']['fees']:>12,.0f} {row['hourly']['reward']:>9,.0f} "
                  f"{row['hourly']['in_range']:>7.1%}   "
                  f"{row['per_swap']['fees']:>14,.0f} {row['per_swap']['reward']:>9,.0f} "
                  f"{row['per_swap']['in_range']:>7.1%}   {lift:>8.2f}x")
            for fm in row:
                if row[fm]["reward"] > best.get(fm, (-np.inf, None))[0]:
                    best[fm] = (row[fm]["reward"], w)
        print("-" * 104)
        for fm in ("hourly", "per_swap"):
            r, w = best[fm]
            print(f"  best width under {fm:<9}: {w:>6.0f} ticks "
                  f"(+/-{1.0001**w-1:.1%}), reward {r:,.0f}")


if __name__ == "__main__":
    main()
