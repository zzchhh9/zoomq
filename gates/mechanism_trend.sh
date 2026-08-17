#!/bin/bash
# Record how ZoomQ's mechanism counters move as training accumulates.
#
#   setsid nohup ./gates/mechanism_trend.sh > logs/mechanism_trend.log 2>&1 < /dev/null &
#
# WHY A TREND AND NOT A SINGLE READING. At 2000 frames the exec_cond_skeleton fix already lifted
# Q^exec's action sensitivity from 0.115 to 0.93 C51 atoms (8x, 4 seeds), but the across-depth
# spread — the quantity the stopping rule actually gates on — stayed at 0.001-0.037 against a
# u_r floor of one atom. Those two numbers moving differently over training is the whole
# question, and one snapshot cannot show it. Each pass appends a row per arm, so the answer is a
# curve rather than an anecdote.
#
# Read-only with respect to training: it loads snapshots on CPU and writes only its own JSON.
# INTERVAL is generous because the probe costs a couple of cores and the box is shared.
set -uo pipefail

cd /mnt/workspace/zoomq
INTERVAL="${INTERVAL:-1800}"
PATTERN="${PATTERN:-p7_mp_*}"
OUT=gates/probe/mechanism_trend.jsonl

set +u; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -u
export HOME=/mnt/workspace/zoomq/demos
export PYTHONUSERBASE=/root/.local
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/mnt/workspace/zoomq/third_party/CQN-AS-G1
PY=/mnt/workspace/anchorq/.venv/bin/python

mkdir -p gates/probe

while true; do
  stamp=$(date -Is)
  tmp=$(mktemp)
  if $PY gates/probe_exec_mechanism.py --glob "$PATTERN" --states 12 --json "$tmp" \
        > /dev/null 2>&1; then
    # Stamp each row with the wall clock and the arm's frame count, so the trend can be
    # plotted against frames rather than against pass number.
    $PY - "$tmp" "$stamp" "$OUT" <<'PYEOF'
import csv, json, os, sys
res, stamp, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
with open(out, "a") as f:
    for r in res:
        p = "/mnt/workspace/zoomq/runs/%s/train.csv" % r.get("run", "")
        frame = None
        if os.path.exists(p):
            try:
                rows = list(csv.DictReader(open(p)))
                if rows:
                    frame = int(float(rows[-1]["frame"]))
            except Exception:
                pass
        r["t"] = stamp
        r["frame"] = frame
        f.write(json.dumps(r) + "\n")
print("appended %d rows" % len(res))
PYEOF
    echo "$stamp ok"
  else
    echo "$stamp probe failed (arms may have no snapshot yet); retrying next pass"
  fi
  rm -f "$tmp"
  sleep "$INTERVAL"
done
