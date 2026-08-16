#!/bin/bash
# Append a monitor snapshot every INTERVAL seconds. The depth histogram over
# time is the actual evidence for "does the stopping rule move as the exec heads
# acquire data" -- a single end-of-night reading cannot show that.
OUT=/mnt/workspace/zoomq/logs/status_series.log
INT=${INTERVAL:-600}
while true; do
  { echo "===== $(date -Iseconds) ====="
    /mnt/workspace/anchorq/.venv/bin/python /mnt/workspace/zoomq/scripts/monitor.py 2>&1
  } >> "$OUT"
  sleep "$INT"
done
