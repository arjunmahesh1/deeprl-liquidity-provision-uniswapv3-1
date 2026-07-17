"""Algorithm registry with a per-algorithm kwarg filter.

The previous config declared one PPO-specific hyperparameter block (`clip_range`,
`target_kl`, `gae_lambda`, `vf_coef`) that no other algorithm accepts, which is why
swapping the algorithm was never tried. Each algorithm here declares only what it
takes, so adding one is a dict entry.

All are SB3-family on the same `Discrete` action space, so the comparison is about
the algorithm rather than the action space. SAC and TD3 are absent deliberately:
they need a `Box` space, which is a different formulation.
"""
from __future__ import annotations

from stable_baselines3 import A2C, DQN, PPO

# Kwargs each algorithm accepts beyond the common set. A kwarg outside its list is
# dropped rather than passed, so a shared config cannot crash one algorithm.
_ACCEPTS = {
    "ppo": {"learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "target_kl", "n_epochs"},
    "a2c": {"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef"},
    "dqn": {"learning_rate", "batch_size", "gamma", "buffer_size", "learning_starts",
            "target_update_interval", "exploration_fraction",
            "exploration_final_eps", "train_freq"},
}

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def make_agent(name: str, env, seed: int, device: str = "cpu", **kwargs):
    """Build an agent, passing only the kwargs its algorithm accepts.

    device defaults to cpu on purpose: the nets are small enough that GPU loses to
    kernel-launch overhead, and MPS cannot run this env at all (it emits float64
    observations and MPS has no float64).
    """
    key = name.lower()
    if key not in ALGOS:
        raise ValueError(f"unknown algorithm {name!r}; have {sorted(ALGOS)}")
    allowed = _ACCEPTS[key]
    dropped = set(kwargs) - allowed
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if dropped:
        # Visible, not silent: a dropped kwarg means the config and the algorithm
        # disagree, and the run should say so.
        print(f"  [{key}] ignoring kwargs it does not accept: {sorted(dropped)}")
    return ALGOS[key]("MlpPolicy", env, seed=seed, device=device, verbose=0, **filtered)
