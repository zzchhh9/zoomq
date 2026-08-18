#!/bin/bash
# Delete the replay buffers of FINISHED arms. Approved by the user.
#
# /mnt/workspace hit 10T of 10T with 0 bytes available, which silently stopped
# every live training arm. A single run dir is ~250 GB and is almost entirely
# replay buffer: for p2_fc_base_s1, buffer/ 142 GB + demo_buffer/ 115 GB against
# snapshot.pt 431 MB and train.csv 40 KB.
#
# WHAT IS LOST: only the ability to RESUME these arms. They are all at ~100,000
# frames, i.e. finished. train.csv, eval results, snapshot.pt, tb/ and
# eval_snapshots/ are untouched, so every published number stays reproducible
# from disk.
#
# SAFETY. Each arm is re-verified immediately before its buffers are removed:
# train.csv must read >= 99000 frames (parsed by HEADER NAME -- `frame` is column
# 23, not 45), snapshot.pt must exist, and no trainer process may hold it. Any
# arm failing a check is skipped and reported, never deleted anyway.
set -uo pipefail
PY=/mnt/workspace/anchorq/.venv/bin/python
[ -x "$PY" ] || PY=python3
RUNS=/mnt/workspace/zoomq/runs

ARMS="p1_mp_s1 p1_mp_s2 p1_mp_s3 p1_sh_s1 p1_sh_s2 p1_sh_s3 \
p2_fc_base_s1 p2_fc_base_s2 p2_pc_base_s1 p2_pc_base_s2 \
p2_tc_base_s1 p2_tc_base_s2 rep_movepl"

echo "=== before ==="; df -h /mnt/workspace | tail -1

for a in $ARMS; do
  d="$RUNS/$a"
  # re-verify, immediately before deleting this arm's buffers
  fr=$($PY - "$d/train.csv" <<'PYEOF' 2>/dev/null || echo 0
import csv, sys
rs = list(csv.DictReader(open(sys.argv[1])))
print(int(float(rs[-1]["frame"])) if rs else 0)
PYEOF
)
  if [ "${fr:-0}" -lt 99000 ] 2>/dev/null; then
    echo "SKIP $a: train.csv reads ${fr:-unreadable} frames (< 99000)"; continue
  fi
  if [ ! -f "$d/snapshot.pt" ]; then
    echo "SKIP $a: snapshot.pt missing -- deleting its buffer would strand it"; continue
  fi
  if pgrep -f "train_cqn_as_bigym.*runs/${a}\b" >/dev/null 2>&1; then
    echo "SKIP $a: a trainer process is holding it"; continue
  fi
  for sub in buffer demo_buffer; do
    if [ -d "$d/$sub" ]; then
      sz=$(timeout 100 du -sh "$d/$sub" 2>/dev/null | cut -f1)
      rm -rf "$d/$sub"
      echo "removed $a/$sub ($sz)"
    fi
  done
done

echo "=== after ==="; df -h /mnt/workspace | tail -1
echo "=== survivors per arm (must still show train.csv + snapshot.pt) ==="
for a in $ARMS; do
  printf "  %-16s csv=%s snap=%s eval=%s tb=%s\n" "$a" \
    "$([ -f $RUNS/$a/train.csv ] && echo ok || echo MISSING)" \
    "$([ -f $RUNS/$a/snapshot.pt ] && echo ok || echo MISSING)" \
    "$([ -e $RUNS/$a/eval.csv ] || [ -d $RUNS/$a/eval_offline ] && echo ok || echo none)" \
    "$([ -d $RUNS/$a/tb ] && echo ok || echo none)"
done
