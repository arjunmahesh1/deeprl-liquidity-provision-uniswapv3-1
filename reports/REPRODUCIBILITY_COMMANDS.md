# Reproducibility commands

These commands use the actual local and Duke CS cluster paths for this project. All
training commands are resumable: completed, correctly keyed JSON units are skipped.
The workloads are CPU-bound; do not request a GPU.

## Local setup and paths

```bash
cd "/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/DeFi/deeprl-liquidity-provision-uniswapv3-1"
conda activate deeprl-uniswap
python -m pytest -q

export REMOTE_HOST="am1015@login.cs.duke.edu"
export REMOTE_PROJECT="/home/users/am1015/deeprl-liquidity-provision-uniswapv3-fixed"
export REMOTE_CONDA="/home/users/am1015/.conda"
./scripts/sync_to_cluster.sh push
```

The ordinary corrected five-algorithm run can also be executed locally or sharded
across machines:

```bash
./scripts/run_algos.sh \
  "ppo a2c dqn qrdqn recurrentppo" outputs/algos_v1
SHARD=0 OF=8 ./scripts/run_algos.sh \
  "ppo a2c dqn qrdqn recurrentppo" outputs/algos_v1
```

## Cluster submission

```bash
ssh am1015@login.cs.duke.edu
cd /home/users/am1015/deeprl-liquidity-provision-uniswapv3-fixed
export PROJECT_ROOT="/home/users/am1015/deeprl-liquidity-provision-uniswapv3-fixed"
export CONDA_PREFIX="/home/users/am1015/.conda/envs/deeprl-uniswap"
mkdir -p slurm_logs

for algo in ppo a2c dqn qrdqn recurrentppo; do
  sbatch --array=0-35 scripts/slurm/rolling_array.sh \
    outputs/algos_v1 "$algo" event_driven none default
done

for algo in ppo a2c dqn qrdqn recurrentppo; do
  sbatch --array=0-35 scripts/slurm/transfer_array.sh \
    outputs/transfer_v1 "$algo"
done

for spec in \
  "hourly none default" \
  "daily none default" \
  "weekly none default" \
  "event_driven shadow default" \
  "event_driven lvr default" \
  "event_driven none paper"
do
  set -- $spec
  sbatch --array=0-35 scripts/slurm/rolling_array.sh \
    outputs/sensitivity_v1 ppo "$1" "$2" "$3"
done

sbatch --array=0-35 scripts/slurm/retail_frontier_array.sh \
  outputs/retail_frontier_v1

for algo in ppo a2c dqn qrdqn recurrentppo; do
  sbatch --array=0-35 scripts/slurm/rolling_array.sh \
    outputs/action_geometry_v1 "$algo" event_driven none default \
    spacing "45 50 55" "480 540 600"
done
```

Monitor only these experiments with:

```bash
squeue -u am1015
find outputs/algos_v1 -maxdepth 1 -name '*.json' | wc -l
find outputs/transfer_v1 -maxdepth 1 -name '*.json' | wc -l
find outputs/sensitivity_v1 -maxdepth 1 -name '*.json' | wc -l
find outputs/retail_frontier_v1 -maxdepth 1 -name '*.json' | wc -l
find outputs/action_geometry_v1 -maxdepth 1 -name '*.json' | wc -l
```

## Pull and audit

Run locally:

```bash
cd "/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/DeFi/deeprl-liquidity-provision-uniswapv3-1"
export REMOTE_HOST="am1015@login.cs.duke.edu"
export REMOTE_PROJECT="/home/users/am1015/deeprl-liquidity-provision-uniswapv3-fixed"
export REMOTE_CONDA="/home/users/am1015/.conda"
./scripts/sync_to_cluster.sh pull

python -m src.deeprl_liquidity_provision_uniswapv3.experiments.validate_results \
  --primary outputs/algos_v1 --transfer outputs/transfer_v1 \
  --sensitivity outputs/sensitivity_v1 --output outputs/final_audit.json

python -m src.deeprl_liquidity_provision_uniswapv3.experiments.validate_retail_frontier \
  --root outputs/retail_frontier_v1 --primary outputs/algos_v1 \
  --output outputs/retail_frontier_v1/audit.json

python -m src.deeprl_liquidity_provision_uniswapv3.experiments.validate_action_geometry \
  --root outputs/action_geometry_v1 \
  --output outputs/action_geometry_v1/audit.json
```

## Dependence-aware aggregation

```bash
for block in 2 4 6 8; do
  python -m src.deeprl_liquidity_provision_uniswapv3.experiments.combined \
    --out outputs/algos_v1 \
    --report-dir "outputs/algos_v1/combined_block${block}" \
    --bootstrap-block "$block"

  for algo in ppo a2c dqn qrdqn recurrentppo; do
    python -m src.deeprl_liquidity_provision_uniswapv3.experiments.transfer_rolling \
      --out outputs/transfer_v1 --algo "$algo" --aggregate \
      --report-dir "outputs/transfer_v1/aggregate_${algo}_block${block}" \
      --bootstrap-block "$block"
  done

  python -m src.deeprl_liquidity_provision_uniswapv3.experiments.sensitivity \
    --primary outputs/algos_v1 --sensitivities outputs/sensitivity_v1 \
    --report-dir "outputs/sensitivity_v1/aggregate_block${block}" \
    --bootstrap-block "$block"

  python -m src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier \
    --out outputs/retail_frontier_v1 --aggregate \
    --report-dir "outputs/retail_frontier_v1/aggregate_block${block}" \
    --bootstrap-block "$block"

  python -m src.deeprl_liquidity_provision_uniswapv3.experiments.action_geometry \
    --paper outputs/algos_v1 --matched outputs/action_geometry_v1 \
    --report-dir "outputs/action_geometry_v1/aggregate_block${block}" \
    --bootstrap-block "$block"
done
```

Every inferential command uses one seed-averaged test window as one observation,
preserves adjacent-window dependence with a seeded circular moving-block bootstrap,
and performs the declared within-pool Holm correction.
