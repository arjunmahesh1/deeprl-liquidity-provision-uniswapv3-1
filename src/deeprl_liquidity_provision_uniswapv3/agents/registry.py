"""Algorithm registry with a per-algorithm kwarg filter.

The previous config declared one PPO-specific hyperparameter block (`clip_range`,
`target_kl`, `gae_lambda`, `vf_coef`) that no other algorithm accepts, which is why
swapping the algorithm was never tried. Each algorithm here declares only what it
takes, so adding one is a dict entry.

All are SB3-family on the same `Discrete` action space, so the comparison is about
the algorithm rather than the action space. SAC and TD3 are absent deliberately:
they need a `Box` space, which is a different formulation.

`net_arch` and `features_extractor` are routed into `policy_kwargs`. The previous
paper did NOT use a default MLP: it built PPO with a `CustomMLPFeatureExtractor`
whose first layer is `nn.BatchNorm1d(obs_dim, affine=False)`, feeding Optuna-selected
hidden layers (`dim_hidden_layers`, default [4, 2]) and a `tanh` activation into a
128-dim output. That BatchNorm is what made a state containing raw prices (~2,000),
raw liquidity (~1e18) and TA-Lib indicators trainable at all. Handing that state to a
default `MlpPolicy` with no input normalization is not the paper's agent.
"""
from __future__ import annotations

from stable_baselines3 import A2C, DQN, PPO

from .extractors import PaperMLPExtractor

# sb3-contrib carries the two the reviews named by name: a distributional DQN and a
# recurrent policy. Optional so a missing install degrades to a clear error on use
# rather than an import failure that takes the whole package down.
try:
    from sb3_contrib import QRDQN, RecurrentPPO
    _CONTRIB = {"qrdqn": QRDQN, "recurrentppo": RecurrentPPO}
except ImportError:  # pragma: no cover
    _CONTRIB = {}

# Kwargs each algorithm accepts beyond the common set. A kwarg outside its list is
# dropped rather than passed, so a shared config cannot crash one algorithm.
ACCEPTS = {
    "ppo": {"learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "target_kl", "n_epochs"},
    "a2c": {"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef"},
    "dqn": {"learning_rate", "batch_size", "gamma", "buffer_size", "learning_starts",
            "target_update_interval", "exploration_fraction",
            "exploration_final_eps", "train_freq"},
    "qrdqn": {"learning_rate", "batch_size", "gamma", "buffer_size", "learning_starts",
              "target_update_interval", "exploration_fraction",
              "exploration_final_eps", "train_freq"},
    "recurrentppo": {"learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda",
                     "clip_range", "ent_coef", "vf_coef", "target_kl", "n_epochs"},
}

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN, **_CONTRIB}

# The policy class each algorithm needs. RecurrentPPO cannot take "MlpPolicy".
_POLICY = {"recurrentppo": "MlpLstmPolicy"}

def make_agent(name: str, env, seed: int, device: str = "cpu",
               net_arch=None, paper_extractor: bool = False, **kwargs):
    """Build an agent, passing only the kwargs its algorithm accepts.

    device defaults to cpu on purpose: the nets are small enough that GPU loses to
    kernel-launch overhead, and MPS cannot run this env at all (it emits float64
    observations and MPS has no float64).

    `paper_extractor=True` reproduces the previous paper's policy network
    (BatchNorm -> net_arch -> 128, tanh). `net_arch` alone changes the hidden layers
    while leaving SB3's default extractor in place.
    """
    key = name.lower()
    if key not in ALGOS:
        extra = "" if _CONTRIB else " (sb3-contrib is not installed: pip install sb3-contrib)"
        raise ValueError(f"unknown algorithm {name!r}; have {sorted(ALGOS)}{extra}")
    allowed = ACCEPTS[key]
    dropped = set(kwargs) - allowed
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if dropped:
        # Visible, not silent: a dropped kwarg means the config and the algorithm
        # disagree, and the run should say so. NOTE for anyone building a grid: a
        # dropped kwarg COLLAPSES configs. A {lr} x {ent_coef} grid handed to DQN,
        # which takes no ent_coef, becomes duplicate runs of the same config, and a
        # selection over it compares a config against itself.
        print(f"  [{key}] ignoring kwargs it does not accept: {sorted(dropped)}")

    pk = {}
    if paper_extractor:
        pk["features_extractor_class"] = PaperMLPExtractor
        pk["features_extractor_kwargs"] = dict(features_dim=128,
                                               hidden_dim=net_arch or [4, 2],
                                               activation="tanh")
    elif net_arch is not None:
        # Every algorithm here takes net_arch through policy_kwargs, QR-DQN included.
        pk["net_arch"] = list(net_arch)

    return ALGOS[key](_POLICY.get(key, "MlpPolicy"), env, seed=seed, device=device,
                      verbose=0, policy_kwargs=pk or None, **filtered)
