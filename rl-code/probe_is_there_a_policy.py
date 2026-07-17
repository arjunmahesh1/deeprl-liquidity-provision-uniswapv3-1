"""Is there any policy that beats passive? If not, there is no RL paper.

PPO converges to never-rebalance because, on the action grid the previous paper
used ([0,45,50,55], all sub-1% bands), passive really is the best option. That grid
was itself selected by maximizing test reward, so it is not evidence about what a
good grid looks like.

Three axes, swept as fixed policies so the answer does not depend on RL working:
  1. width      - how wide should a passive range be?
  2. threshold  - recentre only when price drifts this far out; how far?
  3. capital    - gas is fixed per rebalance, fees scale with size, so scale should
                  decide whether active management pays at all.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS

PANEL = Path(__file__).resolve().parents[1] / "data/processed"
WARMUP = 168
WINDOW = 1500


def make(key, lo, widths, capital, gas=5.0):
    p = POOLS[key]
    df = pd.read_parquet(PANEL / f"{key}_hourly.parquet").iloc[lo:lo + WINDOW].reset_index(drop=True)
    return UniswapV3Env(df, fee_tier_pct=p.fee_tier_pct, action_widths=np.array(widths),
                        dec0=p.dec0, dec1=p.dec1, capital_usd=capital,
                        gas_usd=gas, warmup=WARMUP)


def roll(env, policy):
    obs, _ = env.reset()
    tot, done, trunc = 0.0, False, False
    info = {}
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(policy(env))
        tot += r
    return tot, info


def passive(env):
    return 0


def out_of_range(env):
    """Recentre whenever the price leaves the band."""
    return 0 if env.sqrtA <= env.sqrtP[env.i] <= env.sqrtB else 1


def med(key, starts, widths, capital, policy, gas=5.0):
    rs = [roll(make(key, s, widths, capital, gas), policy)[0] for s in starts]
    return float(np.median(rs))


def main():
    key = "usdc_weth_005"
    starts = [i * WINDOW for i in range(1, 9)]
    print(f"{key}: medians over {len(starts)} windows, capital $30k unless stated\n")

    print("1. PASSIVE at different widths (never rebalance)")
    print(f"   {'width':>8} {'reward':>12}")
    for w in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        print(f"   {w:>8} {med(key, starts, [0, w], 30_000, passive):>12,.0f}")

    print("\n2. RECENTRE WHEN OUT OF RANGE, by width (the wider, the rarer)")
    print(f"   {'width':>8} {'reward':>12}")
    for w in (50, 200, 500, 1000, 2000, 5000):
        print(f"   {w:>8} {med(key, starts, [0, w], 30_000, out_of_range):>12,.0f}")

    print("\n3. SCALE: gas is fixed per rebalance, fees scale with capital")
    print(f"   {'capital':>12} {'passive w=1000':>16} {'active w=1000':>15} {'active-passive':>15}")
    for cap in (30_000, 100_000, 1_000_000, 10_000_000):
        p = med(key, starts, [0, 1000], cap, passive)
        a = med(key, starts, [0, 1000], cap, out_of_range)
        print(f"   {cap:>12,} {p:>16,.0f} {a:>15,.0f} {a-p:>15,.0f}")


if __name__ == "__main__":
    main()
