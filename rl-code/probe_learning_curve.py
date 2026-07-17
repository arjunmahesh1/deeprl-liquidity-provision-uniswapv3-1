"""How many timesteps does PPO actually need here?

Trains once and evaluates on VALIDATION at checkpoints, so we pick the timestep
budget from where reward plateaus instead of guessing. Validation is carved from
the train block; test is never touched, which is the protocol we are moving to.

CAVEAT: this runs on the CURRENT (broken) fee model, where delta=0.05 means a 5%
pool and fee income is ~69% of IL. Once the fee model is fixed the reward changes
shape, so re-run this before freezing the budget. It is still worth doing now for
the order of magnitude.
"""
import sys, time
sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor

WINDOW = 1500
TRAIN_H = 6000          # 4 windows
VAL_H = 1500            # last window of the 7500h block
CHECKPOINTS = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
SEEDS = [42, 123, 256]


def build_env(df, params):
    return Uniswapv3Env(
        delta=params["delta"],
        action_values=np.array(params["action_values"], dtype=float),
        market_data=df,
        x=params["x"],
        gas=params["gas_fee"],
        reward_type="IL",
    )


def evaluate(model, env):
    obs, _ = env.reset(seed=0)
    total, done, trunc = 0.0, False, False
    actions = []
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, r, done, trunc, _ = env.step(a)
        total += r
    # action distribution matters: a collapsed constant policy is not learning
    frac_rebal = float(np.mean([a != 0 for a in actions]))
    return total, frac_rebal


def main():
    with open("config/uniswap_rl_param_1108.yaml") as f:
        params = yaml.safe_load(f)
    prices = pd.read_csv("data_price_uni_h_time.csv")[["price"]].reset_index(drop=True)
    train_df = prices.iloc[:TRAIN_H].reset_index(drop=True)
    val_df = prices.iloc[TRAIN_H:TRAIN_H + VAL_H].reset_index(drop=True)

    print(f"train {len(train_df)}h -> validate {len(val_df)}h (test untouched)")
    print(f"{'steps':>10} " + " ".join(f"{'s'+str(s):>12}" for s in SEEDS)
          + f" {'mean':>12} {'rebal%':>8}")
    print("-" * 76)

    policy_kwargs = dict(
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            activation=params["activation"],
            hidden_dim=params["dim_hidden_layers"],
        ),
    )

    models, val_envs = {}, {}
    for s in SEEDS:
        env = build_env(train_df, params)
        models[s] = PPO(
            "MlpPolicy", Monitor(env),
            learning_rate=params["learning_rate"],
            n_steps=max(1, len(env.market_data) // 3),
            batch_size=params["batch_size"], gamma=params["gamma"],
            gae_lambda=params["gae_lambda"], clip_range=params["clip_range"],
            ent_coef=params["ent_coef"], vf_coef=params["vf_coef"],
            target_kl=params["target_kl"], policy_kwargs=policy_kwargs,
            seed=s, device="cpu", verbose=0,
        )
        val_envs[s] = build_env(val_df, params)

    t0 = time.perf_counter()
    prev = 0
    for ck in CHECKPOINTS:
        row, rebals = [], []
        for s in SEEDS:
            # continue training from where we left off
            models[s].learn(total_timesteps=ck - prev, reset_num_timesteps=False)
            v, fr = evaluate(models[s], val_envs[s])
            row.append(v); rebals.append(fr)
        prev = ck
        print(f"{ck:>10,} " + " ".join(f"{v:>12,.0f}" for v in row)
              + f" {np.mean(row):>12,.0f} {np.mean(rebals)*100:>7.1f}%")

    print(f"\nelapsed {time.perf_counter()-t0:,.0f}s")
    print("Pick the smallest budget where the mean has plateaued.")
    print("rebal% near 0 or 100 = collapsed policy, not learning.")


if __name__ == "__main__":
    main()
