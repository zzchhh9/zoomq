#!/bin/bash
# Sample p12's rolling snapshot.pt so round-0 level-2 can be reconstructed later.
#
#   setsid nohup ./scripts/p12_snapshot_sampler.sh > logs/p12_sampler.log 2>&1 < /dev/null &
#
# WHY THIS EXISTS. scripts/launch_phase12.sh set save_eval_snapshot=false and
# EVAL_EVERY=100000, so p12 writes NO intermediate eval checkpoints and would finish
# with nothing to probe: binacc_r0_L2 and spread_r0_L2 are not train.csv columns, they
# have to be computed from weights. p9 had save_eval_snapshot=true and EVAL_EVERY=2000
# and therefore has a trajectory; p12 as launched would not. That is a defect in the
# launcher, not in the arm, and this recovers from it without restarting the run:
# snapshot.pt IS written continuously (mtime tracks the frontier), so copying it on a
# timer reconstructs the same history after the fact.
#
# It only copies when the frame has advanced, so a stalled arm costs nothing, and it
# stops on its own once the arm is past the decision point.
set -uo pipefail
cd /mnt/workspace/zoomq
OUT=runs/_p12_snaps
mkdir -p "$OUT"
PY=/mnt/workspace/anchorq/.venv/bin/python
INTERVAL="${INTERVAL:-900}"     # 15 minutes
STOP_FRAMES="${STOP_FRAMES:-30000}"   # past the 90-episode cut

frame_of() {
  $PY - "$1" <<'PYEOF' 2>/dev/null || echo 0
import csv, sys
rs = list(csv.DictReader(open(sys.argv[1])))
fr = [int(float(r["frame"])) for r in rs if r.get("frame")]
print(max(fr) if fr else 0)
PYEOF
}

declare -A last
while true; do
  done_all=1
  for a in p12_mp_zqEAw_s1 p12_mp_zqEAw_s2; do
    csv="runs/$a/train.csv"; snap="runs/$a/snapshot.pt"
    [ -f "$csv" ] && [ -f "$snap" ] || continue
    fr=$(frame_of "$csv")
    [ "${fr:-0}" -ge "$STOP_FRAMES" ] || done_all=0
    prev="${last[$a]:-0}"
    if [ "${fr:-0}" -gt "$prev" ]; then
      # avail check: a 460 MB copy must never be the thing that fills the disk again
      avail=$(df -BG --output=avail /mnt/workspace | tail -1 | tr -dc '0-9')
      if [ "${avail:-0}" -lt 300 ]; then
        echo "$(date -Is) SKIP $a: only ${avail} GB free"
        continue
      fi
      cp "$snap" "$OUT/${a}_f$(printf '%07d' "$fr").pt" && last[$a]=$fr
      echo "$(date -Is) sampled $a at frame $fr ($(ls -la "$OUT" | wc -l) files, ${avail} GB free)"
    fi
  done
  [ "$done_all" -eq 1 ] && { echo "$(date -Is) both arms past ${STOP_FRAMES} frames; sampler exiting"; break; }
  sleep "$INTERVAL"
done
