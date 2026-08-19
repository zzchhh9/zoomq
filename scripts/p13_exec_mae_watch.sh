#!/bin/bash
# 30-min exec-MAE watch for p13. One snap per process; retry SIGSEGV.
set -uo pipefail
cd /mnt/workspace/zoomq
LOCK=/mnt/workspace/zoomq/logs/p13_exec_mae_watch.lock
INTERVAL="${INTERVAL:-1800}"
STOP_EP="${STOP_EP:-50}"
PY=/mnt/workspace/anchorq/.venv/bin/python
TICK=./scripts/p13_exec_mae_tick.py

if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "$(date -Is) already running pid=$old"
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

set +eu
# shellcheck disable=SC1091
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
set -eu
export HOME=/mnt/workspace/zoomq/demos
export PYTHONUSERBASE=/root/.local
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONFAULTHANDLER=1

ep_of() {
  $PY - "$1" <<'PY' 2>/dev/null || echo 0
import csv, sys
n = 0
for r in csv.DictReader(open(sys.argv[1])):
    if r.get("episode"):
        n = max(n, int(float(r["episode"])))
print(n)
PY
}

dump_table() {
  $PY - <<'PY' 2>/dev/null || true
import json, os
p = "/mnt/workspace/zoomq/logs/p13_exec_mae.jsonl"
print("%-22s %-16s %8s %8s %7s %s" % ("arm", "tag", "full", "adapt", "ratio", "modal_ad"))
if not os.path.exists(p):
    print("(no jsonl yet)")
    raise SystemExit
for ln in open(p):
    r = json.loads(ln)
    print("%-22s %-16s %8.4f %8.4f %7s %s" % (
        r.get("arm",""), str(r.get("tag",""))[:16],
        r.get("mae_full_depth") or 0, r.get("mae_adaptive") or 0,
        r.get("ratio_ad_over_full"), r.get("modal_adaptive")))
PY
}

run_cycle() {
  for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
    ok=0
    for try in 1 2 3 4; do
      set +e
      $PY "$TICK"
      rc=$?
      set -e
      if [ "$rc" -eq 0 ]; then ok=1; break; fi
      if [ "$rc" -eq 2 ]; then return 0; fi
      echo "$(date -Is) snap-try $n attempt $try rc=$rc"
      sleep 4
    done
    [ "$ok" -eq 1 ] || { echo "$(date -Is) giving up this snap after 4 attempts"; return 1; }
  done
}

echo "$(date -Is) watch start interval=${INTERVAL}s stop_ep=${STOP_EP}"
while true; do
  avail=$(df -BG --output=avail /mnt/workspace | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt 200 ]; then
    echo "$(date -Is) SKIP: ${avail} GB free"
  else
    echo "$(date -Is) --- cycle ---"
    run_cycle || true
    echo "--- table ---"
    dump_table
  fi
  done_all=1
  for a in p13_mp_zqEAd_s1 p13_mp_zqEAd_s2; do
    f=runs/$a/train.csv
    [ -f "$f" ] || { done_all=0; continue; }
    ep=$(ep_of "$f")
    [ "${ep:-0}" -ge "$STOP_EP" ] || done_all=0
  done
  if [ "$done_all" -eq 1 ]; then
    echo "$(date -Is) both arms past ${STOP_EP} ep; last cycle"
    run_cycle || true
    dump_table
    break
  fi
  sleep "$INTERVAL"
done
echo "$(date -Is) watch exit"
