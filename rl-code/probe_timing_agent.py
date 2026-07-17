"""Can a learned policy time liquidity provision better than heuristics?

The reformulated problem: staying out scores 0, always-in loses ~1,265/window, and
an oracle with perfect foresight makes ~1,262. The question is how much of that
2,527 gap anything can close from observable state alone.

Heuristics try the same signal the agent sees, so the comparison is about the
policy rather than the information. Everything is fit and evaluated on VALIDATION;
test is untouched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS

PANEL = Path(__file__).resolve().parents[1] / "data/processed"
W, WARMUP = 1500, 168
WIDTHS = np.array([200.0, 1000.0])
HOLD, EXIT = 0, 1
ENTER = 2
SEEDS = [42, 123, 256]
STEPS = 150_000


def make(key, lo, n_windows=1):
    p = POOLS[key]
    df = pd.read_parquet(PANEL / f"{key}_hourly.parquet").iloc[lo:lo + W * n_windows].reset_index(drop=True)
    return UniswapV3Env(df, fee_tier_pct=p.fee_tier_pct, action_widths=WIDTHS,
                        dec0=p.dec0, dec1=p.dec1, capital_usd=30_000.0,
                        gas_usd=5.0, warmup=WARMUP)


def roll(env, policy):
    obs, _ = env.reset()
    tot, acts, done, trunc = 0.0, [], False, False
    while not (done or trunc):
        a = policy(obs, env)
        acts.append(int(a))
        obs, r, done, trunc, _ = env.step(a)
        tot += r
    return tot, acts


# --- heuristics, all reading the same observable state the agent gets ---------

def always_out(o, e):
    return HOLD


def always_in(o, e):
    return HOLD if e.in_position else ENTER


def vol_threshold(thr):
    """In when realized vol is low: IL grows with volatility, fees do not."""
    def f(o, e):
        quiet = e.vol[e.i] < thr
        if quiet and not e.in_position:
            return ENTER
        if not quiet and e.in_position:
            return EXIT
        return HOLD
    return f


def fee_over_vol(thr):
    """In when the pool pays enough per unit of liquidity relative to volatility."""
    def f(o, e):
        payout = e.fees_usd[e.i] / e.pool_L[e.i] if e.pool_L[e.i] > 0 else 0.0
        signal = payout / (e.vol[e.i] ** 2 + 1e-12)
        rich = signal > thr
        if rich and not e.in_position:
            return ENTER
        if not rich and e.in_position:
            return EXIT
        return HOLD
    return f


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "usdc_weth_005"
    train_lo, val_lo = 0, W * 4
    print(f"{key}: train hours 0-{W*4}, validate {val_lo}-{val_lo+W}\n")

    # Calibrate each heuristic family on TRAIN, then report on VALIDATION.
    def val(pol):
        return roll(make(key, val_lo), pol)[0]

    def tr(pol):
        return roll(make(key, train_lo, 4), pol)[0]

    rows = [("stay out", always_out), ("always in", always_in)]
    vt = max([0.002, 0.005, 0.01, 0.02, 0.05], key=lambda t: tr(vol_threshold(t)))
    rows.append((f"vol < {vt}", vol_threshold(vt)))
    fv = max([1e-14, 1e-13, 1e-12, 1e-11, 1e-10], key=lambda t: tr(fee_over_vol(t)))
    rows.append((f"fee/vol^2 > {fv:.0e}", fee_over_vol(fv)))

    print(f"{'policy':<24} {'val reward':>11} {'actions':>8}")
    print("-" * 46)
    for name, pol in rows:
        r, acts = roll(make(key, val_lo), pol)
        print(f"{name:<24} {r:>11,.0f} {len(set(acts)):>8}")

    print(f"\nPPO, {len(SEEDS)} seeds x {STEPS:,} steps on train, evaluated on validation")
    res = []
    for s in SEEDS:
        m = PPO("MlpPolicy", DummyVecEnv([lambda: Monitor(make(key, train_lo, 4))]),
                learning_rate=3e-4, ent_coef=0.01, seed=s, device="cpu", verbose=0)
        m.learn(total_timesteps=STEPS)
        r, acts = roll(make(key, val_lo), lambda o, e, m=m: m.predict(o, deterministic=True)[0])
        res.append(r)
        print(f"  seed {s:<5} {r:>11,.0f}   actions used: {sorted(set(acts))}")
    print(f"  {'mean':<10} {np.mean(res):>11,.0f}   sd {np.std(res):,.0f}")


if __name__ == "__main__":
    main()
