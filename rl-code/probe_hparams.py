"""Are the Optuna-selected hyperparameters the cause of the policy collapse?

The current config was chosen by maximizing TEST reward over 10 trials, and it
sits at the boundary of its ranges: lr=1e-2 (PPO convention is 3e-4), target_kl=0.3
(convention 0.01-0.05), ent_coef=1e-4. Those are the classic ingredients of policy
saturation. If SB3-default-ish values learn stably where these collapse, then the
hyperparameters are a symptom of the leak, and the fix is published defaults rather
than a per-window search.

Evaluates on VALIDATION only. Test is never touched.

CAVEAT: runs on the current (broken) 5% fee model. Re-check after the fee fix.
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

TRAIN_H, VAL_H = 6000, 1500
CHECKPOINTS = [25_000, 100_000, 250_000]
SEEDS = [42, 123, 256]

CONFIGS = {
    "optuna-selected (current)": dict(
        learning_rate=1e-2, ent_coef=1e-4, target_kl=0.3,
        gamma=0.999, gae_lambda=0.9999, clip_range=0.2, vf_coef=0.1,
        hidden=[4, 2], activation="tanh",
    ),
    "sb3 defaults": dict(
        learning_rate=3e-4, ent_coef=0.0, target_kl=None,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
        hidden=[64, 64], activation="tanh",
    ),
    "sb3 defaults + entropy": dict(
        learning_rate=3e-4, ent_coef=0.01, target_kl=0.05,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
        hidden=[64, 64], activation="tanh",
    ),
}


def build_env(df, base):
    return Uniswapv3Env(
        delta=base["delta"],
        action_values=np.array(base["action_values"], dtype=float),
        market_data=df, x=base["x"], gas=base["gas_fee"], reward_type="IL",
    )


def evaluate(model, env):
    obs, _ = env.reset(seed=0)
    total, done, trunc, acts = 0.0, False, False, []
    while not (done or trunc):
        a, _ = model.predict(obs, deterministic=True)
        acts.append(int(a))
        obs, r, done, trunc, _ = env.step(a)
        total += r
    uniq = len(set(acts))
    return total, float(np.mean([a != 0 for a in acts])), uniq


def main():
    with open("config/uniswap_rl_param_1108.yaml") as f:
        base = yaml.safe_load(f)
    prices = pd.read_csv("data_price_uni_h_time.csv")[["price"]].reset_index(drop=True)
    train_df = prices.iloc[:TRAIN_H].reset_index(drop=True)
    val_df = prices.iloc[TRAIN_H:TRAIN_H + VAL_H].reset_index(drop=True)

    for name, cfg in CONFIGS.items():
        print(f"\n=== {name} ===")
        print(f"    lr={cfg['learning_rate']} ent={cfg['ent_coef']} "
              f"kl={cfg['target_kl']} net={cfg['hidden']}")
        print(f"{'steps':>9} {'mean val':>12} {'spread':>12} {'rebal%':>8} {'#actions':>9}")

        pk = dict(
            features_extractor_class=CustomMLPFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=128, activation=cfg["activation"], hidden_dim=cfg["hidden"]),
        )
        models, envs = {}, {}
        for s in SEEDS:
            env = build_env(train_df, base)
            models[s] = PPO(
                "MlpPolicy", Monitor(env),
                learning_rate=cfg["learning_rate"], n_steps=max(1, len(env.market_data) // 3),
                batch_size=base["batch_size"], gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
                ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"],
                target_kl=cfg["target_kl"], policy_kwargs=pk,
                seed=s, device="cpu", verbose=0,
            )
            envs[s] = build_env(val_df, base)

        prev = 0
        for ck in CHECKPOINTS:
            vals, rebs, uqs = [], [], []
            for s in SEEDS:
                models[s].learn(total_timesteps=ck - prev, reset_num_timesteps=False)
                v, rb, uq = evaluate(models[s], envs[s])
                vals.append(v); rebs.append(rb); uqs.append(uq)
            prev = ck
            print(f"{ck:>9,} {np.mean(vals):>12,.0f} {np.std(vals):>12,.0f} "
                  f"{np.mean(rebs)*100:>7.1f}% {np.mean(uqs):>9.1f}")

    print("\n#actions = distinct actions the deterministic policy uses on validation.")
    print("1.0 means the policy collapsed to a constant and is not controlling anything.")


if __name__ == "__main__":
    main()
