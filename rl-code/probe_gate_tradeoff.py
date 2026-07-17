"""THE GATE: with real volume, is fee income the same order as impermanent loss?

If it is not, the reward is still ~-IL, there is nothing to trade off, and no RL
paper survives. This is the day-5 gate from the plan, answered on the real panel.

Models a passive LP: deposit once at t=0 into a FIXED tick range, never rebalance.
Fees accrue only while the price sits inside that range, at our share of in-range
liquidity. Position value uses the real Uniswap v3 formulas, so impermanent loss
is the concentrated-position loss, not the textbook full-range one.

Everything is computed in RAW units, matching the pool's `liquidity` column. L is
invariant to token relabeling in raw units, which is what makes our_L / pool_L a
legitimate ratio; the share is asserted into (0, 1] regardless.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS

PANEL = Path(__file__).resolve().parents[1] / "data/processed"
Q96 = 2 ** 96
WINDOW = 1500
X_ETH = 10.0
WIDTHS = [45, 50, 55, 100, 200, 1000]


def simulate(df: pd.DataFrame, pool, width: int, start: int):
    """Passive LP over `WINDOW` hours from `start`. Returns (fees, il, in_range_frac, share)."""
    win = df.iloc[start:start + WINDOW]
    sqrtP = win["sqrt_price_x96"].to_numpy() / Q96
    pool_L = win["liquidity"].to_numpy()
    fees_gross = win["fees_usd"].to_numpy()
    price_usd = win["price"].to_numpy()          # USD per WETH
    tick0 = float(win["tick"].iloc[0])

    # Range fixed at deposit and held. This is the whole point: it must not follow price.
    sqrtPa = 1.0001 ** ((tick0 - width) / 2.0)
    sqrtPb = 1.0001 ** ((tick0 + width) / 2.0)
    sqrtP0 = sqrtP[0]

    # token0=USDC(6), token1=WETH(18); P_raw = WETH_raw / USDC_raw.
    # Deposit X_ETH of token1 -> L from amount1 = L(sqrtP - sqrtPa).
    amount1_raw = X_ETH * 10 ** pool.dec1
    L = amount1_raw / max(sqrtP0 - sqrtPa, 1e-30)

    # Matching token0 leg the position needs at deposit.
    amount0_raw = L * (1.0 / sqrtP0 - 1.0 / sqrtPb)

    a0_0 = amount0_raw / 10 ** pool.dec0   # USDC
    a1_0 = amount1_raw / 10 ** pool.dec1   # WETH
    hold_usd = a0_0 * 1.0 + a1_0 * price_usd   # value of just keeping the two tokens

    # Position value through time, real v3 branches.
    below, above = sqrtP < sqrtPa, sqrtP > sqrtPb
    inside = ~below & ~above
    a0 = np.where(below, L * (1.0 / sqrtPa - 1.0 / sqrtPb),
         np.where(above, 0.0, L * (1.0 / np.maximum(sqrtP, 1e-30) - 1.0 / sqrtPb)))
    a1 = np.where(above, L * (sqrtPb - sqrtPa),
         np.where(below, 0.0, L * (sqrtP - sqrtPa)))
    pos_usd = (a0 / 10 ** pool.dec0) * 1.0 + (a1 / 10 ** pool.dec1) * price_usd

    share = np.divide(L, pool_L, out=np.zeros_like(pool_L, dtype=float), where=pool_L > 0)
    share = np.clip(share, 0.0, 1.0)
    fees = float((fees_gross * share * inside).sum())
    il = float(hold_usd[-1] - pos_usd[-1])
    return fees, il, float(inside.mean()), float(np.median(share[inside])) if inside.any() else 0.0


def run(key: str, starts):
    pool = POOLS[key]
    if not (pool.token0 in ("USDC", "USDT") and pool.token1 == "WETH"):
        return
    df = pd.read_parquet(PANEL / f"{key}_hourly.parquet")
    print(f"\n{key}   {pool.pair} @ {pool.fee_tier:.2%}   position {X_ETH} ETH")
    print(f"  {'width':>6} {'share':>8} {'in-range':>9} {'fees $':>10} {'IL $':>10} {'fees/IL':>9}")
    for w in WIDTHS:
        F, I, R, S = zip(*(simulate(df, pool, w, s) for s in starts))
        f, i, r, s = np.median(F), np.median(I), np.mean(R), np.median(S)
        ratio = f / i if i > 0 else np.nan
        print(f"  {w:>6} {s:>7.3%} {r:>8.1%} {f:>10,.0f} {i:>10,.0f} "
              f"{ratio:>8.2f}x" if np.isfinite(ratio) else
              f"  {w:>6} {s:>7.3%} {r:>8.1%} {f:>10,.0f} {i:>10,.0f} {'n/a':>9}")


if __name__ == "__main__":
    print("GATE: fee income vs impermanent loss for a passive LP, real panel")
    print("Medians over 8 windows. A tradeoff needs fees/IL within ~an order of 1.")
    starts = [i * WINDOW for i in range(8)]
    for k in ["usdc_weth_005", "usdc_weth_030", "weth_usdt_005", "weth_usdt_030"]:
        run(k, starts)
