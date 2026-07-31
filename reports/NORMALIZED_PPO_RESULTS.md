# Gated reward-normalized PPO diagnostic

## Status and audit

**COMPLETE — FINAL COLLECTION.**

- Scope: matched USDC/WETH 0.05% and 0.30% pools
- Walk-forward units: 24 windows per pool, 48/48 total
- Cluster array: Slurm job `12254993`
- Structural audit: no missing, duplicate, malformed, or provenance-mismatched units
- Collection SHA-256:
  `0683a410bc93ded0e936281c17b59bcebd6922094a2ef86f9a02b9c00cecd5f5`
- Tests: 177 passed

The gate was opened prospectively because the complete headroom experiment found
168-hour local action disagreement of 67.0%--73.2% across the six pools, with
similarly high 24-hour estimates. The treatment and outcomes were frozen in
`reports/NORMALIZED_PPO_PROTOCOL.md` before the normalized models were trained.

## Treatment

The treatment differs from the completed matched-geometry PPO reference only in
training-reward normalization:

- Stable-Baselines3 `VecNormalize`
- `norm_obs=False`, `norm_reward=True`
- reward clipping at 10
- normalization discount 0.99 and epsilon \(10^{-8}\)

Validation and test bypass the wrapper and use raw dollar rewards. The observation
space, 13 features, reward definition, `{45,50,55}` action labels,
`{480,540,600}` executable raw-tick widths, $30,000 capital, $5 gas, event-driven
schedule, 20,000-step budget, seeds 42 and 123, and hyperparameter-selection
protocol are unchanged.

One inferential observation is one seed-averaged held-out window. The predeclared
H4 family has one paired normalized-minus-standard reward contrast within each
pool; therefore its Holm-adjusted p-value equals its raw p-value.

## Primary results

Primary inference uses 10,000 seeded circular moving-block bootstrap resamples and
block length four.

| Pool | Standard PPO mean | Normalized PPO mean | Normalized − standard (95% CI) | Holm p | Norm. wins |
|---|---:|---:|---:|---:|---:|
| USDC/WETH 0.05% | −$1,096 | −$1,989 | −$893 [−$1,275, −$562] | .0001 | 3/24 |
| USDC/WETH 0.30% | −$891 | −$947 | −$56 [−$742, +$503] | .8651 | 7/24 |

Reward normalization significantly **worsens** the low-fee pool and produces no
detectable reward change in the high-fee pool.

## Mechanistic diagnostics

| Pool | Standard multi-action windows | Normalized multi-action windows | Entropy 0.01 selected |
|---|---:|---:|---:|
| USDC/WETH 0.05% | 0/24 | 0/24 | 11/24 (45.8%) |
| USDC/WETH 0.30% | 0/24 | 0/24 | 8/24 (33.3%) |

Normalization never changes the number of distinct deterministic test actions:
both arms use exactly one action in every window. Positive entropy is selected by
validation in a substantial minority of windows, but it does not yield
state-dependent deterministic behavior.

## Dependence robustness

| Pool | Block 2 | Block 4 (primary) | Block 6 | Block 8 |
|---|---|---|---|---|
| USDC/WETH 0.05% | −$893 [−$1,407, −$427], p=.0003 | −$893 [−$1,275, −$562], p=.0001 | −$893 [−$1,178, −$582], p=.0001 | −$893 [−$1,149, −$607], p=.0001 |
| USDC/WETH 0.30% | −$56 [−$779, +$532], p=.8681 | −$56 [−$742, +$503], p=.8651 | −$56 [−$714, +$512], p=.8619 | −$56 [−$675, +$478], p=.8525 |

The conclusion is invariant to all predeclared block lengths.

## Interpretation for the paper

The headroom result rules out the convenient claim that PPO collapses because
conditional opportunity is absent. This gated result additionally rules out raw
reward scale as a sufficient explanation under the tested remedy: reward-only
normalization plus validation-selected entropy does not produce state-dependent
control and can reduce held-out economic reward.

This is a bounded diagnostic, not proof that every stabilization method must fail.
It tests one standard normalization treatment on the matched USDC/WETH fee-tier
pair under the unchanged CLEB budget and features. Larger budgets, recurrent
normalization schemes, alternative reward shaping, or richer causal policies remain
open, but they cannot be claimed as explanations for the present benchmark.

## Reproducibility

Machine-readable outputs:

- `outputs/normalized_ppo_v1/audit.json`
- `outputs/normalized_ppo_v1/aggregate_block4/windows.csv`
- `outputs/normalized_ppo_v1/aggregate_block4/summary.csv`
- `outputs/normalized_ppo_v1/aggregate_block4/report.txt`
- corresponding `aggregate_block2`, `aggregate_block6`, and `aggregate_block8`
  directories

The exact commands are recorded in `reports/REPRODUCIBILITY_COMMANDS.md`.
