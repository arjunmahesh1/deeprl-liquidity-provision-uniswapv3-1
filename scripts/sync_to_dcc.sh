#!/bin/bash
# Push code and the processed panel to DCC. Pull results back.
#
# Needs the multiplexed SSH master to be open, which needs a real terminal because
# Duke enforces Duo MFA even with keys. Open it once with `ssh dcc` in Terminal
# (NetID password, then 1 for a Duo push); it persists for 8h and everything here
# reuses it without another prompt.
#
#   ./scripts/sync_to_dcc.sh push      code + data/processed  (~541MB the first time)
#   ./scripts/sync_to_dcc.sh pull      outputs/ back here
#   ./scripts/sync_to_dcc.sh status    queue + how many work units have landed
#
# The raw swap parquet in ~/Projects/defi-rv is NOT synced: it is 4.9GB and the
# cluster only needs the aggregated panel that data/aggregate.py and data/swaps.py
# already produced from it.

set -euo pipefail

REMOTE=dcc
PROJECT=/hpc/group/darec/ab978/deeprl-liquidity-provision-uniswapv3
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# macOS ships openrsync (protocol 29), which has no --info=progress2 and no
# --exclude-from niceties. Prefer a real rsync if Homebrew put one on the PATH,
# otherwise fall back to flags openrsync actually understands.
RSYNC_FLAGS="-az"
if rsync --version 2>/dev/null | head -1 | grep -qv openrsync; then
  RSYNC_FLAGS="-az --info=progress2"
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" true 2>/dev/null; then
  echo "cannot reach $REMOTE. The SSH master is closed or expired (>8h)." >&2
  echo "Open it yourself in a terminal:  ssh dcc   (NetID password, then 1 for Duo)" >&2
  exit 1
fi

case "${1:-}" in
  push)
    ssh "$REMOTE" "mkdir -p $PROJECT/slurm_logs $PROJECT/outputs $PROJECT/data/processed"
    echo "--- code ---"
    rsync $RSYNC_FLAGS \
      --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'data' --exclude 'outputs' --exclude 'rl-code/output' \
      --exclude 'paper' --exclude '.DS_Store' \
      "$LOCAL/" "$REMOTE:$PROJECT/"
    echo "--- panel (only what changed) ---"
    rsync $RSYNC_FLAGS "$LOCAL/data/processed/" "$REMOTE:$PROJECT/data/processed/"
    echo "--- env check ---"
    ssh "$REMOTE" "test -x /hpc/group/darec/ab978/miniconda3/envs/deeprl-uniswap/bin/python3 \
      && echo 'env deeprl-uniswap: present' \
      || echo 'env deeprl-uniswap: MISSING -> conda env create -f $PROJECT/environment.yml'"
    ;;
  pull)
    mkdir -p "$LOCAL/outputs"
    rsync $RSYNC_FLAGS "$REMOTE:$PROJECT/outputs/" "$LOCAL/outputs/"
    ;;
  status)
    ssh "$REMOTE" "squeue -u ab978 -o '%.10i %.9P %.20j %.2t %.10M %.6D %R' | head -20; \
      echo '--- work units landed ---'; \
      for d in $PROJECT/outputs/*/; do \
        [ -d \"\$d\" ] && echo \"\$(basename \$d): \$(ls \$d/*.json 2>/dev/null | wc -l) units\"; \
      done"
    ;;
  *)
    echo "usage: $0 {push|pull|status}" >&2
    exit 2
    ;;
esac
