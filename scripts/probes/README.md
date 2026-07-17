# Probes

Diagnostics that run against the CURRENT code (`src/`). They lived under `rl-code/`,
which every document fences off as "do not build on this", so anyone following that
advice was ignoring working tools.

| script | what it answers |
|---|---|
| `probe_gate_tradeoff.py` | Is there a real fee/IL tradeoff on the panel, or is the reward just -IL? |
| `probe_is_there_a_policy.py` | Is there headroom a policy could capture at all? |
| `probe_new_env.py` | Does the rebuilt environment behave sanely end to end? |
| `probe_timing_agent.py` | How fast does an agent step, for sizing a sweep? |

Run from the repo root with the project env active.

The probes that target the OLD environment (`probe_fee_units`, `probe_hparams`,
`probe_learning_curve`, `probe_timing`) stay under `rl-code/`: they import
`custom_env_folder.custom_env` and only make sense against that code.
