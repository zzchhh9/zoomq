#!/bin/bash
# The phase-12 gate. Two rules it exists to enforce, both learned the hard way.
#
#   ./scripts/p12_gate.sh
#
# RULE 1 -- COMPARE AT MATCHED FRAMES. eps_depth anneals 0.3 -> 0.05 over 20000
# steps, so depth_share_0 and the clamp rates move on their own. At 1200 frames
# p12_mp_zqEAw_s1 and p9_mp_zqEA_s1 report depth_share_0 = 0.7384001855055491,
# bit-identical -- while p9 at 25401 frames reads 0.8876 purely because eps reached
# 0.05. An earlier version of this script printed p12@1200 beside p9@25401 and read
# the difference as a treatment effect. It is not. Same for the clamp: matched at
# 1200 frames, window_clamp_rate_4 is 0.01550 (p9) vs 0.00172 (p12), a 9.0x drop --
# not the 16x that the mismatched pair suggested.
#
# RULE 2 -- spread_r0_L2 AND binacc_r0_L2 ARE NOT IN train.csv. They come from a
# snapshot probe. save_eval_snapshot=false for these arms, but snapshot.pt is live
# and gates/../critic_localize.py reads it, so the column is obtainable -- it just
# has to be computed, not looked up.
#
# WHY THOSE TWO COLUMNS DECIDE THE NULL. p12 widens rounds 1-4 and leaves ROUND 0 at
# w=1.0. The stopping rule is dead (delta_over_kappa_u p90 = 0.0000), so rollouts
# execute the round-0 two-knot fill: success lives on round 0, the treatment lives on
# rounds 1-4. A flat round-0 level 2 with a pretty deep-round one is a BLIND null.
# Reference: zqEA spread_r0_L2 0.0027 / binacc_r0_L2 0.494; CQN-AS L2 4.42 / 0.916;
# chance 0.20.
set -uo pipefail
cd /mnt/workspace/zoomq
PY=/mnt/workspace/anchorq/.venv/bin/python

echo "=== 1. successes at the pre-registered cut (first 45 episodes) ==="
$PY - <<'PYEOF'
import csv, glob, os
groups = [("baseline stock", "runs/p1_mp_s*"), ("baseline stock", "runs/rep_movepl"),
          ("zqEA (control)", "runs/p9_mp_zqEA_s*"),
          ("zqEAw (p12)", "runs/p12_mp_zqEAw_s*")]
print("  %-16s %-20s %5s %8s %8s  %s" % ("group", "arm", "eps", "succ@45", "succ@90", "first success at ep"))
pooled = {}
for tag, pat in groups:
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "train.csv")
        if not os.path.exists(f):
            continue
        ep = [float(r["episode_reward"]) for r in csv.DictReader(open(f))
              if r.get("episode_reward") not in (None, "")]
        nz = [i + 1 for i, x in enumerate(ep) if x > 0]
        s45 = sum(1 for x in ep[:45] if x > 0) if len(ep) >= 45 else None
        s90 = sum(1 for x in ep[:90] if x > 0) if len(ep) >= 90 else None
        print("  %-16s %-20s %5d %8s %8s  %s" % (tag, os.path.basename(d), len(ep),
              s45 if s45 is not None else "-", s90 if s90 is not None else "-",
              nz[0] if nz else "-"))
        if s45 is not None:
            pooled.setdefault(tag, []).append(s45)
        tag = ""
print()
for k, v in pooled.items():
    if len(v) >= 2:
        print("  pooled@45  %-16s %d / %d" % (k, sum(v), 45 * len(v)))
print()
print("  RULE vs zqEA's own 11/90:  >=20 lattice was binding | <=11 null | 12-19 inconclusive")
print("  A single early success is NOT a signal -- zqEA's own first lands at episode 20-21.")
PYEOF

echo
echo "=== 2. did the treatment reach rounds 1-4? MATCHED FRAMES ONLY ==="
$PY - <<'PYEOF'
import csv, glob, os
COLS = ["depth_share_0", "window_clamp_rate_1", "window_clamp_rate_2",
        "window_clamp_rate_3", "window_clamp_rate_4", "delta_over_kappa_u_p90"]
def row_near(f, target, tol=400):
    best = None
    for r in csv.DictReader(open(f)):
        try: fr = int(float(r.get("frame") or 0))
        except ValueError: continue
        if abs(fr - target) <= tol and (best is None or abs(fr - target) < abs(best[0] - target)):
            best = (fr, r)
    return best

arms = sorted(glob.glob("runs/p12_mp_zqEAw_s*")) + sorted(glob.glob("runs/p9_mp_zqEA_s*"))
frames = []
for d in arms:
    f = os.path.join(d, "train.csv")
    if os.path.exists(f):
        rs = [int(float(r["frame"])) for r in csv.DictReader(open(f)) if r.get("frame")]
        if rs: frames.append(max(rs))
target = min(frames) if frames else 1200
print("  comparing every arm at ~%d frames (the least-advanced arm's frontier)" % target)
print("  %-20s %7s %s" % ("arm", "frame", " ".join("%14s" % c[-14:] for c in COLS)))
for d in arms:
    f = os.path.join(d, "train.csv")
    if not os.path.exists(f): continue
    b = row_near(f, target)
    if not b:
        print("  %-20s   (no row near %d)" % (os.path.basename(d), target)); continue
    fr, r = b
    def fmt(c):
        # Format as a float. String-truncating these once chopped the "e-05" off
        # delta_over_kappa_u_p90 and made 1.27e-05 read as 1.27 -- i.e. a dead
        # stopping rule reading as one that fires.
        v = r.get(c)
        if v in (None, ""):
            return "-"
        try:
            x = float(v)
        except ValueError:
            return str(v)[:14]
        return ("%14.4g" % x).strip()
    print("  %-20s %7d %s" % (os.path.basename(d), fr,
          " ".join("%14s" % fmt(c) for c in COLS)))
print()
print("  depth_share_0 and the clamp rates are eps-driven; they are ONLY comparable")
print("  between arms at the same frame. Never read them across different frames.")
PYEOF

echo
echo "=== 3. round-0 level 2 -- computed from snapshot.pt, not looked up in train.csv ==="
PROBE=/root/gap/critic_localize.py
if [ ! -f "$PROBE" ]; then
  echo "  MISSING $PROBE -- rebuild it before reading this gate; the success count alone"
  echo "  cannot tell a real null from a treatment that never reached round 0."
  exit 1
fi
for a in p12_mp_zqEAw_s1 p12_mp_zqEAw_s2; do
  s="runs/$a/snapshot.pt"
  if [ -f "$s" ]; then
    printf "  %-20s snapshot.pt %s bytes, mtime %s\n" "$a" \
      "$(stat -c %s "$s")" "$(stat -c %y "$s" | cut -c1-19)"
  else
    printf "  %-20s snapshot.pt MISSING\n" "$a"
  fi
done
echo "  run:  set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -eu"
echo "        HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local OMP_NUM_THREADS=8 \\"
echo "        $PY $PROBE   # point it at runs/p12_mp_zqEAw_s1/snapshot.pt"
echo "  reference: zqEA spread_r0_L2 0.0027 / binacc_r0_L2 0.494 | CQN-AS 4.42 / 0.916 | chance 0.20"
