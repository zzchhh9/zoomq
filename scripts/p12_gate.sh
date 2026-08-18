#!/bin/bash
# The phase-12 gate, instrumented so a null is READABLE rather than blind.
#
#   ./scripts/p12_gate.sh
#
# WHY THE EXTRA COLUMNS. p12 widens rounds 1-4 (w_schedule 1.0/0.2/0.2/0.2/0.2) but
# leaves ROUND 0 at w=1.0 -- and the stopping rule is dead (delta_over_kappa_u p90
# 0.0000, depth_share_1..4 all sitting on the annealed eps floor), so training
# rollouts execute the round-0 two-knot fill. Success therefore lives on round 0
# while the treatment lives on rounds 1-4. Chunk MAE averages all rounds, so the
# treatment can open deep-round level 2, drop MAE, and still print zqEA's 4,7.
#
# So the success count alone cannot tell "the lattice was not the constraint" from
# "the treatment never reached the thing that decides success". These four do:
#   spread_r0_L2_atoms  did round 0's finest level open at all?   (zqEA: 0.0027)
#   binacc_r0_L2        can it pick the demo's bin there?         (zqEA: 0.494,
#                       CQN-AS 0.916, chance 0.20)
#   window_clamp_rate_* did widening actually change the clamping?
#   depth_share_0       is round 0 still what executes?           (zqEA: ~0.89)
# A pretty level 2 with a flat round 0 is a blind null, not a clean one.
set -uo pipefail
cd /mnt/workspace/zoomq
PY=/mnt/workspace/anchorq/.venv/bin/python

echo "=== 1. successes at the pre-registered cut (first 45 episodes) ==="
$PY - <<'PYEOF'
import csv, glob, os
groups = [("baseline stock", "runs/p1_mp_s*"), ("", "runs/rep_movepl"),
          ("zqEA (control)", "runs/p9_mp_zqEA_s*"),
          ("zqEAw (p12)", "runs/p12_mp_zqEAw_s*")]
print("  %-22s %-20s %5s %8s %8s" % ("group", "arm", "eps", "succ@45", "succ@90"))
pooled = {}
for tag, pat in groups:
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "train.csv")
        if not os.path.exists(f):
            continue
        ep = [float(r["episode_reward"]) for r in csv.DictReader(open(f))
              if r.get("episode_reward") not in (None, "")]
        s45 = sum(1 for x in ep[:45] if x > 0) if len(ep) >= 45 else None
        s90 = sum(1 for x in ep[:90] if x > 0) if len(ep) >= 90 else None
        print("  %-22s %-20s %5d %8s %8s" % (tag, os.path.basename(d), len(ep),
              s45 if s45 is not None else "-", s90 if s90 is not None else "-"))
        if s45 is not None and tag:
            pooled.setdefault(tag, []).append(s45)
        tag = ""
print()
for k, v in pooled.items():
    if len(v) >= 2:
        print("  pooled@45  %-18s %d / %d" % (k, sum(v), 45 * len(v)))
print()
print("  RULE (vs zqEA's own 11/90):  >=20 lattice was binding | <=11 null | 12-19 inconclusive")
PYEOF

echo
echo "=== 2. did the treatment reach anything? (train.csv, by header name) ==="
$PY - <<'PYEOF'
import csv, glob, os
COLS = ["depth_share_0", "window_clamp_rate_1", "window_clamp_rate_2",
        "window_clamp_rate_3", "window_clamp_rate_4", "delta_over_kappa_u_p90",
        "zoomq_refine_loss", "zoomq_exec_loss"]
def last(rs, c):
    for r in reversed(rs):
        v = r.get(c, "")
        if v not in ("", None):
            try: return float(v)
            except ValueError: return None
    return None
print("  %-20s %8s %s" % ("arm", "frame", " ".join("%18s" % c[:18] for c in COLS)))
for d in sorted(glob.glob("runs/p12_mp_zqEAw_s*")) + sorted(glob.glob("runs/p9_mp_zqEA_s*")):
    f = os.path.join(d, "train.csv")
    if not os.path.exists(f):
        continue
    rs = list(csv.DictReader(open(f)))
    fr = last(rs, "frame")
    vals = [last(rs, c) for c in COLS]
    print("  %-20s %8d %s" % (os.path.basename(d), int(fr or 0),
          " ".join("%18s" % (("%.4f" % v) if v is not None else "-") for v in vals)))
PYEOF

echo
echo "=== 3. round-0 level-2, the column the success count cannot substitute for ==="
echo "    (needs the snapshot probe; reference zqEA spread_r0_L2 0.0027 / binacc_r0_L2 0.494,"
echo "     CQN-AS L2 4.42 / 0.916, chance 0.20)"
if [ -f /root/gap/critic_localize.py ]; then
  echo "    run:  OMP_NUM_THREADS=8 $PY /root/gap/critic_localize.py  (point it at runs/p12_mp_zqEAw_s1/snapshot.pt)"
else
  echo "    /root/gap/critic_localize.py is gone -- rebuild it before reading the gate"
fi
