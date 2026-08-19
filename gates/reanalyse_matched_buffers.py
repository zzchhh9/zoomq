#!/usr/bin/env python3
"""Re-derive every ZoomQ conclusion with MATCHED demonstration buffers.

Every comparison in phases 11-13 used p9_mp_zqEA as the control. p9 crashed and
restarted 15 times and each restart re-appends the whole demonstration set to the
same demo_buffer/, so zqEA trained on 4.3x the data of the arms it was the control
for. This regroups every arm by its actual buffer size and only compares within a
tier.

Also switches the readout from 45 episodes to 90. First successes across this family
land between episode 11 and 54, so a 45-episode cut is inside the noise: p13_s1's
first success is at episode 43 and it scores 8 in episodes 46-60.
"""
import csv, glob, os, subprocess, itertools, math

RUNS = "/mnt/workspace/zoomq/runs"

def restarts(arm):
    n = len(glob.glob(os.path.join(RUNS, arm, "demo_buffer", "*")))
    return n

def series(arm):
    f = os.path.join(RUNS, arm, "train.csv")
    if not os.path.exists(f):
        return None
    rs = list(csv.DictReader(open(f)))
    ep = [float(r["episode_reward"]) for r in rs
          if r.get("episode_reward") not in (None, "")]
    fr = max((int(float(r["frame"])) for r in rs if r.get("frame")), default=0)
    return ep, fr

def cut(ep, k):
    return sum(1 for x in ep[:k] if x > 0) if len(ep) >= k else None

ARMS = {
    "CQN-AS baseline":        ["p1_mp_s1", "p1_mp_s2", "p1_mp_s3", "rep_movepl"],
    "zqEA  (A+B, adaptive)":  ["p9_mp_zqEA_s1", "p9_mp_zqEA_s2"],
    "zqEB  (B only)":         ["p9_mp_zqEB_s1", "p9_mp_zqEB_s2"],
    "zqE   (neither)":        ["p9_mp_zqE_s1", "p9_mp_zqE_s2"],
    "zqES  (A only)":         ["p9_mp_zqES_s1", "p9_mp_zqES_s2"],
    "zqEAT (A+B+tiebreak)":   ["p11_mp_zqEAT_s1", "p11_mp_zqEAT_s2"],
    "zqTR  (tiebreak rand)":  ["p11_mp_zqTR_s1", "p11_mp_zqTR_s2"],
    "zqTM  (tiebreak mid)":   ["p11_mp_zqTM_s1", "p11_mp_zqTM_s2"],
    "zqEAw (A+B+wide win)":   ["p12_mp_zqEAw_s1", "p12_mp_zqEAw_s2"],
    "zqEAd (A+B+full depth)": ["p13_mp_zqEAd_s1", "p13_mp_zqEAd_s2"],
    "zqFull (chain+depth)":   ["p10_mp_zqFull_s1", "p10_mp_zqFull_s2"],
}

print("%-24s %-18s %5s %6s %5s %5s %6s %6s %s" % (
    "group", "arm", "demoF", "eps", "@45", "@90", "@135", "first", "per-15"))
tiers = {}
for label, arms in ARMS.items():
    g = label
    for a in arms:
        s = series(a)
        if not s:
            continue
        ep, fr = s
        nf = restarts(a)
        nz = [i + 1 for i, x in enumerate(ep) if x > 0]
        w = [sum(1 for x in ep[i:i + 15] if x > 0) for i in range(0, min(len(ep), 135), 15)]
        print("%-24s %-18s %5d %6d %5s %5s %6s %6s %s" % (
            g, a, nf, len(ep),
            cut(ep, 45) if cut(ep, 45) is not None else "-",
            cut(ep, 90) if cut(ep, 90) is not None else "-",
            cut(ep, 135) if cut(ep, 135) is not None else "-",
            nz[0] if nz else "-", w))
        tier = "5x (15 restarts)" if nf > 200 else "1x"
        tiers.setdefault((label, tier), []).append((a, cut(ep, 90), nf))
        g = ""

print()
print("=== ONLY 1x-BUFFER ARMS, at 90 episodes ===")
one = {}
for (g, tier), v in tiers.items():
    if tier != "1x":
        continue
    got = [x[1] for x in v if x[1] is not None]
    if got:
        one[g] = got
for g, v in sorted(one.items(), key=lambda kv: -sum(kv[1]) / max(1, len(kv[1]))):
    print("  %-24s per-seed %-14s pooled %d / %d   rate %.4f" % (
        g, str(v), sum(v), 90 * len(v), sum(v) / (90.0 * len(v))))

print()
print("=== seed-permutation test: zqEAd (full depth) vs the adaptive=true 1x family ===")
treat = one.get("zqEAd (A+B+full depth)", [])
ctrl = one.get("zqEAT (A+B+tiebreak)", []) + one.get("zqEAw (A+B+wide win)", [])
if treat and ctrl:
    pool = treat + ctrl
    k = len(treat)
    obs = sum(treat) / len(treat) - sum(ctrl) / len(ctrl)
    ge = tot = 0
    for c in itertools.combinations(range(len(pool)), k):
        t = [pool[i] for i in c]
        r = [pool[i] for i in range(len(pool)) if i not in c]
        d = sum(t) / len(t) - sum(r) / len(r)
        tot += 1
        if d >= obs:
            ge += 1
    print("  treatment %s  control %s" % (treat, ctrl))
    print("  observed mean difference %+.2f successes/90 eps" % obs)
    print("  permutation P(diff >= observed) = %d/%d = %.3f" % (ge, tot, ge / tot))
    print("  (the earlier Poisson-over-episodes P of 2e-04 assumed episodes are")
    print("   exchangeable; variance lives at the SEED -- zqEB is 0/85 and 18/93")
    print("   on one identical config and one identical buffer)")
