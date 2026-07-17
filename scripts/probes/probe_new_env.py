"""Does the rebuilt environment give the agent something to learn?

Runs fixed policies on the real panel to establish what the reward looks like, then
trains PPO and watches the action distribution. The old environment drove PPO to a
constant never-rebalance policy scoring a tenth of random, under every config; if
that still happens, fees were not the problem.

Validation only. Test is never touched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS

PANEL = Path(__file__).resolve().parents[1] / "data/processed"
TRAIN_H, VAL_H, WARMUP = 6000, 1500, 168
WIDTHS = np.array([0, 45, 50, 55])
CHECKPOINTS = [25_000, 100_000, 250_000]
SEEDS = [42, 123, 256]


def make_env(key, lo, hi):
    p = POOLS[key]
    df = pd.read_parquet(PANEL / f"{key}_hourly.parquet").iloc[lo:hi].reset_index(drop=True)
    return UniswapV3Env(df, fee_tier_pct=p.fee_tier_pct, action_widths=WIDTHS,
                        dec0=p.dec0, dec1=p.dec1, capital_usd=30_000.0,
                        gas_usd=5.0, warmup=WARMUP)


def rollout(env, policy):
    obs, _ = env.reset()
    total, acts, done, trunc = 0.0, [], False, False
    while not (done or trunc):
        a = policy(obs, env)
        acts.append(int(a))
        obs, r, done, trunc, info = env.step(a)
        total += r
    return total, acts, info


def describe(name, total, acts, info):
    print(f"  {name:<22} reward {total:>12,.0f}   fees {info['cum_fees']:>10,.0f}  "
          f"gas {info['cum_gas']:>7,.0f}  rebal {info['n_rebalances']:>5}  "
          f"actions {len(set(acts))}")


def main():
    key = "usdc_weth_005"
    print(f"{key}: fixed policies on validation (hours {TRAIN_H}-{TRAIN_H+VAL_H})\n")

    rng = np.random.default_rng(0)
    policies = {
        "passive (never rebal)": lambda o, e: 0,
        "always rebal w=50": lambda o, e: 2,
        "random": lambda o, e: rng.integers(0, len(WIDTHS)),
        "rebal only if out": lambda o, e: (0 if e.sqrtA <= e.sqrtP[e.i] <= e.sqrtB else 2),
    }
    for name, pol in policies.items():
        env = make_env(key, TRAIN_H, TRAIN_H + VAL_H)
        describe(name, *rollout(env, pol))

    print(f"\nPPO learning curve (validation), {len(SEEDS)} seeds")
    print(f"  {'steps':>9} {'mean reward':>14} {'spread':>12} {'actions':>8} {'rebal':>7}")
    models = {}
    for s in SEEDS:
        models[s] = PPO("MlpPolicy", Monitor(make_env(key, 0, TRAIN_H)),
                        learning_rate=3e-4, ent_coef=0.01, seed=s,
                        device="cpu", verbose=0)
    prev = 0
    for ck in CHECKPOINTS:
        rs, us, rb = [], [], []
        for s in SEEDS:
            models[s].learn(total_timesteps=ck - prev, reset_num_timesteps=False)
            env = make_env(key, TRAIN_H, TRAIN_H + VAL_H)
            t, acts, info = rollout(env, lambda o, e, m=models[s]: m.predict(o, deterministic=True)[0])
            rs.append(t); us.append(len(set(acts))); rb.append(info["n_rebalances"])
        prev = ck
        print(f"  {ck:>9,} {np.mean(rs):>14,.0f} {np.std(rs):>12,.0f} "
              f"{np.mean(us):>8.1f} {np.mean(rb):>7.0f}")

    print("\nactions=1 means the policy collapsed to a constant, as the old env always did.")


if __name__ == "__main__":
    main()
