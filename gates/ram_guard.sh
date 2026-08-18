#!/bin/bash
# Free RAM by stopping FINISHED arms before the OOM killer picks live ones at random.
#
#   setsid ./gates/ram_guard.sh > logs/ram_guard.log 2>&1 < /dev/null &
#
# WHY. An earlier OOM episode cost 13 arms, and the kernel chose six that had 40K+
# frames of progress over a dozen sitting at zero. Choosing deliberately is
# strictly better. Projection at the time of writing: 34 live arms need ~1904 GB
# to all reach 100K against 1800 GB physical, so the box is ~100 GB short at the
# far end -- but 8 of those arms (p1/p2 baselines) are ALREADY AT ~100,700 frames
# and are no longer growing. Stopping a finished arm is free: its snapshots,
# train.csv and eval results are all on disk and nothing is lost.
#
# It only ever stops arms that have reached DONE_FRAMES, and only when available
# memory is below LOW_GB. It stops ONE per pass so the effect can be observed
# before more is taken, and it never touches an arm below the frame threshold no
# matter how tight memory gets -- an arm still short of its target is exactly what
# should NOT be sacrificed.
set -uo pipefail

cd /mnt/workspace/zoomq
LOW_GB="${LOW_GB:-260}"
DONE_FRAMES="${DONE_FRAMES:-99000}"
INTERVAL="${INTERVAL:-300}"
PY=/mnt/workspace/anchorq/.venv/bin/python

while true; do
  avail=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${avail:-9999}" -lt "$LOW_GB" ]; then
    # Oldest-finished first, so the arm that has been done longest goes first.
    victim=$($PY - "$DONE_FRAMES" <<'PYEOF'
import csv, glob, os, subprocess, sys
done_at = int(sys.argv[1])
best = None
for d in sorted(glob.glob("/mnt/workspace/zoomq/runs/p[1-9]_*")):
    a = os.path.basename(d)
    if subprocess.run(["pgrep", "-f", r"train_cqn_as_bigym.*runs/%s\b" % a],
                      capture_output=True).returncode != 0:
        continue
    p = os.path.join(d, "train.csv")
    if not os.path.exists(p):
        continue
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        continue
    if not rows:
        continue
    f = float(rows[-1]["frame"])
    if f >= done_at and (best is None or f > best[1]):
        best = (a, f)
print(best[0] if best else "")
PYEOF
)
    if [ -n "$victim" ]; then
      pids=$(pgrep -f "train_cqn_as_bigym.*runs/${victim}\b" || true)
      echo "$(date -Is) avail=${avail}GB < ${LOW_GB}; stopping FINISHED arm $victim ($(printf '%s\n' $pids | grep -c .) pids)"
      [ -n "$pids" ] && kill $pids 2>/dev/null
    else
      echo "$(date -Is) avail=${avail}GB < ${LOW_GB} but NO finished arm to stop -- not touching anything in progress"
    fi
  fi
  sleep "$INTERVAL"
done
