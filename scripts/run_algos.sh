#!/bin/bash
# Run the walk-forward protocol for several algorithms, then print one table each.
#
#   ./scripts/run_algos.sh                          # all five, all six pools
#   ./scripts/run_algos.sh "ppo a2c"                # a subset
#   ./scripts/run_algos.sh "ppo" outputs/my_run     # a named run directory
#   SHARD=0 OF=8 ./scripts/run_algos.sh             # one eighth of the work
#
# Each (algorithm, pool, rolling step) is an independent work unit that writes its own
# JSON and is skipped if already done, so this is resumable and shardable: stop it with
# ctrl-C, rerun, and it picks up. The config keys the filename, so several algorithms
# can share one output directory without colliding.
#
# On DCC use the array wrapper instead, which maps $SLURM_ARRAY_TASK_ID onto the same
# --shard index:  sbatch scripts/slurm/rolling_array.sh
set -euo pipefail

ALGOS=${1:-"ppo a2c dqn qrdqn recurrentppo"}
OUT=${2:-outputs/algos_v1}
SHARD=${SHARD:-0}
OF=${OF:-1}
STEPS=${STEPS:-20000}
SEEDS=${SEEDS:-"42 123"}
POOLS=${POOLS:-""}          # empty = the six core pools

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python}
MOD=src.deeprl_liquidity_provision_uniswapv3.experiments.rolling

if [ ! -d data/processed ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
  echo "No panel in data/processed/." >&2
  echo "Unzip the data bundle here, or build it: see README.md 'Getting the data'." >&2
  exit 1
fi

pool_args=""
[ -n "$POOLS" ] && pool_args="--pools $POOLS"

for algo in $ALGOS; do
  echo ""
  echo "=============================================================="
  echo " $algo   shard $SHARD/$OF   $STEPS steps   seeds $SEEDS"
  echo "=============================================================="
  # shellcheck disable=SC2086
  $PY -m $MOD --algo "$algo" --out "$OUT" --shard "$SHARD" --of "$OF" \
      --steps "$STEPS" --seeds $SEEDS $pool_args
done

# Only the last shard to finish should report, so a sharded run does not print six
# partial tables. With --of 1 there is only one shard and this always runs.
if [ "$OF" = "1" ]; then
  for algo in $ALGOS; do
    echo ""
    echo "### $algo"
    # shellcheck disable=SC2086
    $PY -m $MOD --algo "$algo" --out "$OUT" --aggregate $pool_args
  done
fi
