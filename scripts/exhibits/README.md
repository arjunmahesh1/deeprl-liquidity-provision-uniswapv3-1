# Exhibit scripts

Each produced a result cited in SPEC.md. Per the figures-need-a-committed-source
rule, they live here rather than in /tmp, so any number in the paper can be
regenerated end to end.

| script | what it establishes |
|---|---|
| `exhibit_leak_decomposition.py` | The rejected paper's result, reproduced and decomposed: the leak bought +675, the test-selected action grid handicapped competitors by -2,140, and the sign flips from -1,456 to +1,358. |
| `exhibit_fee_scaling_knife_edge.py` | **The open defect.** Doubling fee income flips the optimum from a degenerate +/-170% band to a realistic +/-5% one. Everything rests on fee income being right within 2x, and hourly discretization biases against concentration. |
| `exhibit_legacy_vs_compact_features.py` | The paper's 13-feature state is worth +381 to PPO and does not close the gap to the rules. |
| `exhibit_ppo_convergence.py` | PPO plateaus by 20k agent steps; the budget is not a strawman. |
| `exhibit_schedule_binds.py` | The decision schedule binds on hourly-acting policies and is a no-op on act-when-out ones. |
| `exhibit_timing_unpredictable.py` | Pre-window features do not predict a window's realized reward: sign accuracy at or below chance on 5 of 6 pools. |

Run from the repo root with the project interpreter:
    /Users/alessiobrini/anaconda3/envs/deeprl-uniswap/bin/python scripts/exhibits/<name>.py
