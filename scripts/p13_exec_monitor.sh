#!/bin/bash
# Watch the two curves that tell you where p13 is going, without waiting 45 episodes.
#
#   setsid nohup ./scripts/p13_exec_monitor.sh > logs/p13_exec_monitor.log 2>&1 < /dev/null &
#
# WHAT IT WATCHES, and why these two and not the scoreboard.
#
# Every "action quality" number quoted so far -- zqEA 0.0210, CQN-AS 0.0078 -- is the
# error of the FULL argmax chunk. But the stopping rule is dead
# (delta_over_kappa_u p90 ~1.3e-05 against the 1.0 it needs, and depth_adaptive
# measures 0.0 on every snapshot of every arm), so descend(adaptive=True) returns the
# ROUND-0 SKELETON: two knots and a linear fill, with the other fourteen computed and
# discarded. Measured on the same weights, that costs
#   zqEA @25401   full 0.0210 -> executed 0.0329   (1.56x)
#   p12  @18.5K   full 0.0196 -> executed 0.0320   (1.63x)
# p13 sets adaptive=false, so it executes the full-depth chunk instead. Hence:
#
#   mae_full_depth   the error of what p13 actually executes. Should track zqEA's own
#                    trajectory downward (zqEA was 0.0832 at 2100 frames and 0.0217 at
#                    24201, so p13's current 0.081-0.091 at 2100-3300 is exactly on it).
#   ratio            mae_adaptive / mae_full_depth. It is ~1.00 on p13 today because an
#                    untrained critic emits nearly the same chunk at every depth. If it
#                    climbs toward the 1.6 seen on trained arms, the deeper rounds are
#                    supplying information that zqEA throws away -- which is the whole
#                    premise of the arm. If it stays at 1.0, they are not, and p13's
#                    treatment is inert regardless of what the scoreboard says.
#
# Neither number is in train.csv; both need the weights. p13 writes eval_snapshots
# every 2000 frames (EVAL_EVERY=2000, save_eval_snapshot=true), so no sampler is
# needed here -- unlike p12, which shipped with save_eval_snapshot=false.
set -uo pipefail
cd /mnt/workspace/zoomq
INTERVAL="${INTERVAL:-1800}"
STOP_FRAMES="${STOP_FRAMES:-30000}"
PY=/mnt/workspace/anchorq/.venv/bin/python
TREND=logs/p13_exec_trend.tsv

[ -f "$TREND" ] || printf "ts\trun\tframe\tmae_full_depth\tmae_adaptive\tratio\n" > "$TREND"

frame_of() {
  $PY - "$1" <<'PYEOF' 2>/dev/null || echo 0
import csv, sys
rs = list(csv.DictReader(open(sys.argv[1])))
fr = [int(float(r["frame"])) for r in rs if r.get("frame")]
print(max(fr) if fr else 0)
PYEOF
}

while true; do
  set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -u
  HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local MUJOCO_GL=egl \
    OMP_NUM_THREADS=8 NT=8 N=128 \
    "$PY" /root/fast/exec_mae.py > /root/fast/exec_mae.log 2>&1 || true

  $PY - "$TREND" <<'PYEOF'
import json, os, re, sys, time
trend = sys.argv[1]
p = "/root/fast/exec_mae.json"
if not os.path.exists(p):
    raise SystemExit
rows = json.load(open(p))
ts = time.strftime("%Y-%m-%dT%H:%M:%S")
seen = set()
if os.path.exists(trend):
    for l in open(trend):
        f = l.split("\t")
        if len(f) > 2:
            seen.add((f[1], f[2]))
with open(trend, "a") as fh:
    for r in rows:
        name = r["run"]
        m = re.search(r"_(\d+)$", name)
        fr = m.group(1) if m else "live"
        if (name, fr) in seen and fr != "live":
            continue          # eval snapshots are immutable; only re-log `live`
        fh.write("%s\t%s\t%s\t%.5f\t%.5f\t%.4f\n" % (
            ts, name, fr, r.get("mae_full_depth", float("nan")),
            r.get("mae_adaptive", float("nan")),
            r.get("ratio_adaptive_over_full") or float("nan")))
print("appended")
PYEOF

  echo "$(date -Is) probe done; trend has $(( $(wc -l < "$TREND") - 1 )) rows"
  done_all=1
  for a in p13_mp_zqEAd_s1 p13_mp_zqEAd_s2; do
    fr=$(frame_of "runs/$a/train.csv")
    echo "   $a at ${fr:-0} frames"
    [ "${fr:-0}" -ge "$STOP_FRAMES" ] || done_all=0
  done
  [ "$done_all" -eq 1 ] && { echo "$(date -Is) both p13 arms past ${STOP_FRAMES}; monitor exiting"; break; }
  sleep "$INTERVAL"
done
