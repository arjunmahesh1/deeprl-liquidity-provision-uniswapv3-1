"""Day-1 sizing probe: how long is one training?

The whole sweep plan (how many algorithms, seeds, trials; what fits on Arjun's
laptop vs DCC) hangs on this number, so measure it instead of guessing.

Also checks the CPU-vs-MPS question: dim_hidden_layers is [4,2], so the nets are
tiny and MPS may well lose to CPU on kernel-launch overhead.
"""
import sys, time, os
sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor

WINDOW = 1500
TIMESTEPS = int(os.environ.get("TIMESTEPS", 10_000))  # scale up from the measured rate


def build_env(df, params):
    return Uniswapv3Env(
        delta=params["delta"],
        action_values=np.array(params["action_values"], dtype=float),
        market_data=df,
        x=params["x"],
        gas=params["gas_fee"],
        reward_type="IL",
    )


def time_training(device, train_df, params):
    env = build_env(train_df, params)
    policy_kwargs = dict(
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            activation=params["activation"],
            hidden_dim=params["dim_hidden_layers"],
        ),
    )
    model = PPO(
        "MlpPolicy",
        Monitor(env),
        learning_rate=params["learning_rate"],
        n_steps=max(1, len(env.market_data) // 3),
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        target_kl=params["target_kl"],
        policy_kwargs=policy_kwargs,
        seed=42,
        device=device,
        verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=TIMESTEPS)
    return time.perf_counter() - t0


def main():
    with open("config/uniswap_rl_param_1108.yaml") as f:
        params = yaml.safe_load(f)
    df = pd.read_csv("data_price_uni_h_time.csv")
    prices = df[["price"]].reset_index(drop=True)
    train_df = prices.iloc[:WINDOW * 5].reset_index(drop=True)  # 7500h train block

    print(f"timing {TIMESTEPS:,} timesteps on a {len(train_df):,}h train block")
    print(f"net: hidden={params['dim_hidden_layers']} features_dim=128 "
          f"(tiny -> expect CPU to win)\n")

    results = {}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        try:
            elapsed = time_training(dev, train_df, params)
        except TypeError as e:
            # The env emits float64 observations and MPS has no float64 support,
            # so MPS is not merely slower here, it cannot run at all without
            # casting the observation space to float32. Not worth it: the nets
            # are [4,2] and CPU wins regardless.
            print(f"{dev:>4}: UNUSABLE -> {str(e).splitlines()[0]}")
            continue
        rate = TIMESTEPS / elapsed
        results[dev] = (elapsed, rate)
        print(f"{dev:>4}: {elapsed:7.1f}s  ({rate:8,.0f} steps/s)")

    best = min(results, key=lambda d: results[d][0])
    print(f"\nfastest usable device: {best}")

    # Extrapolate the real training length and the full grid.
    rate = results[best][1]
    for real_steps in (100_000, 1_000_000):
        per_train = real_steps / rate
        print(f"\nat {real_steps:,} timesteps/training: {per_train:,.0f}s each")
        for n_algos in (3, 5):
            grid = 4 * n_algos * 10 * 5 * 10  # pools x algos x windows x seeds x trials
            core_h = grid * per_train / 3600
            print(f"   {n_algos} algos -> {grid:,} trainings = {core_h:,.0f} core-hours"
                  f"  | 8-core laptop: {core_h/8:,.1f}h  | 64-core DCC: {core_h/64:,.1f}h")


if __name__ == "__main__":
    main()
