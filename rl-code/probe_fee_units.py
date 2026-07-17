"""Fee-unit sanity check, with the reward decomposed.

Runs a passive (never-rebalance) 10 ETH position over several 1500-hour windows
under both readings of `delta`, and splits the reward into fee income and
impermanent loss. Two questions:

  1. Which reading of `delta` is the code carrying?  (fee income scales ~100x)
  2. At the TRUE tier, is fee income even the same order as IL?  If it is not,
     the paper's "balance fee income against impermanent loss" premise has
     nothing to balance.

`delta` is overloaded: _fee_to_tickspacing keys on it as a percent (0.05 ->
spacing 10) while _calculate_fee uses it as a fraction. Passing the true
fraction raises in the lookup, so we construct with the tier key and then set
the fee rate. Decoupling those two is what the real fix must make permanent.
"""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from custom_env_folder.custom_env import Uniswapv3Env

DATA = "data_price_uni_h_time.csv"
WINDOW = 1500
X_ETH = 10.0
TIER_KEY = 0.05  # keys tick spacing = 10; NOT the fee rate


def run_passive(fee_rate, df):
    """Deposit once, never rebalance. Returns (reward, fees, n_steps)."""
    env = Uniswapv3Env(
        delta=TIER_KEY,
        action_values=np.array([0, 45, 50, 55], dtype=float),
        market_data=df,
        x=X_ETH,
        gas=5,
        reward_type="IL",
    )
    env.delta = fee_rate  # spacing already fixed from TIER_KEY
    env.reset(seed=0)
    total = 0.0
    done = truncated = False
    steps = 0
    while not (done or truncated):
        _, r, done, truncated, _ = env.step(0)  # 0 = never rebalance
        total += r
        steps += 1
    return total, float(env.cumul_fee), steps


def main():
    df = pd.read_csv(DATA)
    prices = df[["price"]].reset_index(drop=True)
    n_windows = len(prices) // WINDOW

    print(f"passive 10 ETH position, {WINDOW}h windows, no rebalancing")
    print(f"{'win':>3} {'price move':>11} {'fee@5%':>12} {'fee@0.05%':>11} "
          f"{'IL':>12} {'fee/IL @0.05%':>14}")
    print("-" * 70)

    ratios = []
    for w in range(min(n_windows, 8)):
        win = prices.iloc[w * WINDOW:(w + 1) * WINDOW].reset_index(drop=True)
        p0, pT = float(win["price"].iloc[0]), float(win["price"].iloc[-1])

        r_hi, fee_hi, _ = run_passive(0.05, win)
        r_lo, fee_lo, _ = run_passive(0.0005, win)

        # reward = fees - IL - gas; gas = 0 here (action 0 never pays gas)
        il = fee_lo - r_lo
        ratio = fee_lo / abs(il) if il != 0 else float("nan")
        ratios.append(ratio)

        print(f"{w:>3} {(pT/p0-1)*100:>10.1f}% {fee_hi:>12,.0f} {fee_lo:>11,.0f} "
              f"{il:>12,.0f} {ratio*100:>13.1f}%")

    print("-" * 70)
    print(f"fee inflation factor (5% / 0.05%): {fee_hi/fee_lo:,.1f}x")
    print(f"median fee/IL at the true tier   : {np.nanmedian(ratios)*100:.1f}%")
    print()
    print("If fee income is a small fraction of IL at the true tier, the reward")
    print("is essentially -IL, and the fee/IL tradeoff the paper studies is not")
    print("representable under a price-displacement fee model with no volume.")


if __name__ == "__main__":
    main()
