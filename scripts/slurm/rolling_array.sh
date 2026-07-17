#!/bin/bash
#SBATCH --job-name=uniswap-rolling
#SBATCH --partition=common
#SBATCH --requeue
#SBATCH --array=0-17
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/roll_%A_%a.out
#SBATCH --error=slurm_logs/roll_%A_%a.err
#
# Walk-forward rolling windows, as a SLURM array.
#
# This is a THIN wrapper. It maps $SLURM_ARRAY_TASK_ID onto the same work-unit index
# a laptop passes with `--shard i --of n`, and runs the identical entry point. There
# is no cluster-only code path, so a laptop shard and a cluster shard are the same
# computation and their outputs merge without reconciliation.
#
#   --array=0-17  with  --of 18   =>  18 shards, 8 work units each on the core panel
#
# CPU, not GPU: measured at ~6,100 steps/s on CPU, and the nets are far too small for
# a GPU to beat kernel-launch overhead. The env also emits float64, which MPS cannot
# take at all. Do not move this to gpu-common; there is nothing to gain and it burns
# the allocation.
#
# Usage, from the project root on DCC:
#   mkdir -p slurm_logs
#   sbatch scripts/slurm/rolling_array.sh outputs/rolling_v1
#   # then, once the array drains:
#   python -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
#          --out outputs/rolling_v1 --aggregate
#
# Resumable: a unit whose JSON already exists is skipped, so a requeued or preempted
# task picks up where it stopped rather than redoing the shard.

# NOTE: the #SBATCH lines above are COMMENTS. sbatch parses them itself and the shell
# never expands them, so a ${VAR} there is a literal and the job dies on an unwritable
# log path. They must stay relative (resolved against the submit directory) or be
# hardcoded. Only the lines below this point are shell.
set -euo pipefail

# Set these for your own account, or export them before sbatch:
#   PROJECT_ROOT  the project checkout on the cluster
#   CONDA_PREFIX  the conda env (conda activate sets it for you)
PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the deeprl-uniswap env before sbatch}/bin/python3
OUT=${1:-outputs/rolling_v1}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-18}

cd "$PROJECT"

# Absolute paths: a compute node's working directory is not the project root, and the
# panel is resolved relative to the package, so this must be right before anything runs.
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; build the env from environment.yml first" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data/processed" ]; then
  echo "no panel at $PROJECT/data/processed; run scripts/sync_to_cluster.sh first" >&2
  exit 1
fi

echo "host $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}  out ${OUT}"

srun "$PYTHON" -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
  --out "$OUT" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --of "${N_SHARDS}" \
  --steps 20000 \
  --seeds 42 123 \
  --schedule event_driven \
  --shaping none
