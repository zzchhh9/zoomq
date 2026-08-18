#!/bin/bash
# The phase-10 decision readout. Successes come from train.csv episode_reward,
# not from offline evals -- p3_mp_ctl was culled before its first eval was ever
# read while the answer sat in this column the whole time.
#
# Parse by HEADER NAME. `frame` is column 23, not 45; an earlier guard read
# total_time (97991 s) against a frame threshold (99000) and never fired.
cd /mnt/workspace/zoomq
/mnt/workspace/anchorq/.venv/bin/python - <<'PY'
import csv, glob, os
CUTS = (10, 45, 90, 160)
def row(d):
    f = os.path.join(d, "train.csv")
    if not os.path.exists(f):
        return None
    rs = list(csv.DictReader(open(f)))
    ep = [float(r["episode_reward"]) for r in rs
          if r.get("episode_reward") not in (None, "")]
    fr = max([int(float(r["frame"])) for r in rs if r.get("frame")] or [0])
    return ep, fr
print("%-22s %6s %7s %s" % ("arm", "eps", "frame",
      "".join("%8s" % ("succ@%d" % c) for c in CUTS)))
groups = [("baseline stock", sorted(glob.glob("runs/p1_mp_s*")) + ["runs/rep_movepl"]),
          ("p10 zqFull (stock proto)", sorted(glob.glob("runs/p10_mp_zqFull_*"))),
          ("p10 zqE2 (commit, E=2)", sorted(glob.glob("runs/p10_mp_zqE2_*"))),
          ("p10 ctl (new seeds)", sorted(glob.glob("runs/p10_mp_ctl_*"))),
          ("p8 ctl (old seeds)", sorted(glob.glob("runs/p8_mp_ctl_*"))),
          ("p9 zoomq (stock proto)", sorted(glob.glob("runs/p9_mp_*")))]
for name, ds in groups:
    print("--- %s" % name)
    for d in ds:
        r = row(d)
        if not r:
            continue
        ep, fr = r
        cells = "".join("%8s" % (sum(1 for x in ep[:c] if x > 0) if len(ep) >= c else "-")
                        for c in CUTS)
        print("%-22s %6d %7d %s" % (os.path.basename(d), len(ep), fr, cells))
PY
