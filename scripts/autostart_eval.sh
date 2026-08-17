#!/bin/bash
# Wait for the first snapshots, then start the evaluation daemons once.
# Detached watcher: the training arms write a snapshot every 10k frames and the
# first one lands ~2.5 h in, so polling from an interactive session is wasteful.
cd /mnt/workspace/zoomq
while true; do
  n=$(find runs -path "*eval_snapshots/*.pt" 2>/dev/null | wc -l)
  if [ "$n" -ge 3 ]; then
    echo "$(date -Iseconds) snapshots=$n -> starting eval daemons"
    ./scripts/start_eval_daemons.sh
    exit 0
  fi
  sleep 120
done
