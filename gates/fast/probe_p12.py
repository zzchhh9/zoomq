#!/usr/bin/env python3
"""Test the pre-registered p12 prediction NOW instead of in four hours.

PREDICTION, written before this ran (scripts/launch_phase12.sh header and the
bin/residual threshold model):
  * deep rounds open   -- r4's bin/residual is 0.231, above the measured 0.152
                          threshold; r3 0.142 is borderline; r1 0.049 and r2 0.089
                          stay below
  * ROUND 0 does NOT   -- its bin/residual is 0.246, identical to CQN-AS's, yet w
                          = 0.2 / 0.5 / 1.0 all leave r0's L2 at 0.001-0.006 atoms
Round 0 is what executes 79-89% of the time, so the prediction is: critic metrics
improve, success does not.

Reference: zqEA spread_r0_L2 0.0027 / binacc_r0_L2 0.494; CQN-AS L2 4.42 / 0.916;
chance 0.20.

Reuses /root/gap/critic_localize.py's verified machinery rather than reimplementing
it; only the snapshot list changes. Imports stay lazy -- importing cqn_utils or
bigym_src.zoomq at module scope before the first encoder forward segfaults this
interpreter inside the conv at cqn_as.py:97.
"""
import glob, json, os, re, sys
sys.path.insert(0, "/root/gap")


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("NT", "8")))
    torch.manual_seed(0)
    import critic_discrim as cd
    import critic_localize as cl

    rgb, low, act = cd.build_batch(int(os.environ.get("N", "128")))
    rgb = torch.as_tensor(rgb).float()
    low = torch.as_tensor(low)
    act = torch.as_tensor(act)

    snaps = []
    for p in sorted(glob.glob("/mnt/workspace/zoomq/runs/_p12_snaps/*.pt")):
        m = re.search(r"(p12_mp_zqEAw_s\d)_f(\d+)\.pt$", os.path.basename(p))
        if m:
            snaps.append(("%s_%d" % (m.group(1), int(m.group(2))), p))
    # the control, at comparable frames
    snaps.append(("ZQ_zqEA_25401_control", "/mnt/workspace/zoomq/runs/p9_mp_zqEA_s1/snapshot.pt"))

    out = []
    for name, p in snaps:
        if not os.path.exists(p):
            continue
        try:
            out.append(cl.run(name, p, rgb, low, act))
        except Exception:
            import traceback
            traceback.print_exc()
    json.dump(out, open("/root/fast/p12_localize.json", "w"), indent=2)
    print("DONE %d snapshots" % len(out), flush=True)


if __name__ == "__main__":
    main()
