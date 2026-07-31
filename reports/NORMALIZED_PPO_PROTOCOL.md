# Gated diagnostic: reward-normalized PPO

Gate opened on 2026-07-28 after the complete state-headroom collection was audited
and the prospectively specified block-four analysis was run.

## Gate evidence

The mean within-window 168-hour clairvoyant action-disagreement fraction ranges from
67.0% to 73.2% across the six pools. The 24-hour robustness estimates range from
67.4% to 73.2%. Every upper 95% block-bootstrap confidence limit is far above the
pre-specified 10% no-headroom threshold at both horizons. Local full-horizon
opportunity is therefore non-negligible, so the conditional optimization diagnostic
in `STATE_HEADROOM_PROTOCOL.md` is activated.

## Question

Does normalizing the scale of the training reward prevent PPO from collapsing to a
single deterministic action and improve held-out economic reward?

## Frozen design

- Scope is the matched USDC/WETH pair at 0.05% and 0.30%, chosen because it varies
  fee tier while holding the asset pair fixed. It was chosen before running this
  treatment, not from the headroom ranking.
- Twenty-four held-out walk-forward windows per pool, with the unchanged
  4-train/1-validation/1-test split.
- Same 13 legacy features, corrected per-swap fees, value-conserving rebalancing,
  reward, $30,000 capital, $5 gas, no exit, event-driven schedule, and matched
  `{480,540,600}` raw-tick execution widths as the completed geometry collection.
- Same 20,000 training steps, seeds 42 and 123, learning rates
  `{3e-4,1e-3}`, networks `[4,2]` and `[64,64]`, and entropy coefficients
  `{0,0.01}`. Configuration selection uses validation only, followed by a
  train-plus-validation refit and one test read.
- Treatment uses Stable-Baselines3 `VecNormalize` with `norm_obs=False`,
  `norm_reward=True`, `clip_reward=10`, `gamma=0.99`, and `epsilon=1e-8`.
  Thus only the training reward is normalized. The state is unchanged.
- Evaluation bypasses the normalization wrapper and reports the original raw dollar
  reward from the unshaped environment. Normalization statistics never update on
  validation or test.
- The reference is the already-completed matched-geometry PPO arm with the identical
  grid and protocol but no reward normalization.

## Outcomes and inference

The primary contrast is normalized PPO minus standard PPO held-out dollar reward,
paired by pool and test window. One observation is one seed-averaged window.
Inference uses the seeded circular moving-block bootstrap with 10,000 resamples,
seed 20260728, primary block length four, and dependence checks at blocks two, six,
and eight. The reward contrast is a one-member H4 family within each pool, so its
Holm-adjusted p-value equals its raw p-value.

The mechanistic diagnostic is the number of distinct deterministic actions used by
the first seeded policy on each held-out window, matching the existing CLEB logging
convention. We report the fraction of windows using more than one action and the
selected entropy coefficient. These are descriptive diagnostics, not additional
null-hypothesis families.

The collection is final only at 48/48 `(pool, rolling step)` units. Every result will
be reported regardless of whether normalization improves reward or action diversity.
