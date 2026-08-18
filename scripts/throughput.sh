#!/bin/bash
# Measure frames/s per arm over a real interval. Never read a rate off one
# snapshot -- an earlier "8x" claim was really 2.1x for exactly that reason.
#   ./scripts/throughput.sh <seconds>
cd /mnt/workspace/zoomq
W="${1:-300}"
PY=/mnt/workspace/anchorq/.venv/bin/python
snap() {
  $PY - <<'PY'
import csv, glob, os
for d in sorted(glob.glob("runs/p9_mp_*") + glob.glob("runs/p10_mp_*") +
                glob.glob("runs/p8_mp_*")):
    f = os.path.join(d, "train.csv")
    if not os.path.exists(f):
        continue
    try:
        rs = list(csv.DictReader(open(f)))
    except Exception:
        continue
    fr = max([int(float(r["frame"])) for r in rs if r.get("frame")] or [0])
    ep = sum(1 for r in rs if r.get("episode_reward") not in (None, ""))
    print("%s %d %d" % (os.path.basename(d), fr, ep))
PY
}
snap > /tmp/tp_a.txt
sleep "$W"
snap > /tmp/tp_b.txt
$PY - "$W" <<'PY'
import sys
W = float(sys.argv[1])
a = {l.split()[0]: (int(l.split()[1]), int(l.split()[2]))
     for l in open("/tmp/tp_a.txt") if l.strip()}
b = {l.split()[0]: (int(l.split()[1]), int(l.split()[2]))
     for l in open("/tmp/tp_b.txt") if l.strip()}
print("interval %.0f s" % W)
print("%-22s %8s %8s %7s %8s" % ("arm", "frame", "d_frame", "fps", "episodes"))
tot = 0.0
for k in sorted(b):
    if k not in a:
        print("%-22s %8d %8s %7s %8d  (new)" % (k, b[k][0], "-", "-", b[k][1]))
        continue
    d = b[k][0] - a[k][0]
    fps = d / W
    tot += fps
    print("%-22s %8d %8d %7.3f %8d" % (k, b[k][0], d, fps, b[k][1]))
print("\nfleet total %.2f frames/s" % tot)
PY
