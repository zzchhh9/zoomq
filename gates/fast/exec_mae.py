#!/usr/bin/env python3
"""What p13 actually changes: the error of the action that gets EXECUTED.

Every quality number so far -- chunk MAE 0.0210 for zqEA, 0.0078 for CQN-AS -- is
the error of the FULL argmax chunk. But with the stopping rule dead
(delta_over_kappa_u p90 ~1.3e-05 against the 1.0 it needs), descend(adaptive=True)
returns the ROUND-0 SKELETON: two knots and a linear fill. The other fourteen knots
are computed and thrown away. So the executed action carries the chunk's own knot
error PLUS the skeleton reconstruction error, and nobody has measured the sum.

p13 sets adaptive=false, which executes the full 16-knot chunk instead. Same
weights, same forward pass, same FLOPs -- only the selection changes. This measures
both on the same snapshot, which is the quantity that should predict whether p13
moves the scoreboard.

Reference: skeleton_rmse_0 (train.csv, the demo's own round-0 skeleton error) is
0.0652, so if the two error sources were independent the executed error under
adaptive=True would be ~sqrt(0.021^2 + 0.065^2) = 0.068 -- inside the band where
every arm ever run scored 0-1 -- against 0.021 at full depth.

Read-only. Lazy imports (module-scope cqn_utils/zoomq segfaults the interpreter).
"""
import glob, json, os, re, sys
sys.path.insert(0, "/root/gap")


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("NT", "8")))
    torch.manual_seed(0)
    import critic_discrim as cd

    rgb, low, act = cd.build_batch(int(os.environ.get("N", "128")))
    rgb = torch.as_tensor(rgb).float()
    low = torch.as_tensor(low)
    act = torch.as_tensor(act)

    snaps = []
    for a in ("p13_mp_zqEAd_s1", "p13_mp_zqEAd_s2"):
        for p in sorted(glob.glob("/mnt/workspace/zoomq/runs/%s/eval_snapshots/*.pt" % a)):
            m = re.search(r"snapshot_(\d+)\.pt$", p)
            snaps.append(("%s_%s" % (a, m.group(1) if m else "?"), p))
        p = "/mnt/workspace/zoomq/runs/%s/snapshot.pt" % a
        if os.path.exists(p):
            snaps.append(("%s_live" % a, p))
    for a in ("p9_mp_zqEA_s1", "p12_mp_zqEAw_s1"):
        p = "/mnt/workspace/zoomq/runs/%s/snapshot.pt" % a
        if os.path.exists(p):
            snaps.append(("%s_control" % a, p))

    out = []
    for name, path in snaps:
        try:
            ag = cd.load(path)
            enc = ag.encoder.float().cpu().eval()
            cr = ag.critic.float().cpu().eval()
            kappa = float(getattr(ag, "zq_kappa", 1.0))
            mr = int(getattr(ag, "zq_max_round", cr.num_rounds - 1))
            row = {"run": name,
                   "cfg_adaptive": bool(getattr(ag, "zq_adaptive", True)),
                   "kappa": kappa, "max_round": mr}
            with torch.no_grad():
                # act() encodes BEFORE descending (zoomq.py:1262
                # `rgb_obs = self.encoder(rgb_obs)`), so descend takes features,
                # not raw pixels. Passing raw pixels fails with a shape error.
                feat = torch.cat([enc(rgb[i:i+16]) for i in range(0, rgb.shape[0], 16)])
                for tag, ad in (("full_depth", False), ("adaptive", True)):
                    errs, depths = [], []
                    for i in range(0, rgb.shape[0], 16):
                        a_hat, info = cr.descend(
                            feat[i:i+16], low[i:i+16], max_round=mr,
                            adaptive=ad, kappa=kappa, early_exit=True)
                        e = (a_hat.view(a_hat.shape[0], -1) - act[i:i+16].view(a_hat.shape[0], -1))
                        errs.append(e.abs().mean(-1))
                        d = info.get("depth") if isinstance(info, dict) else None
                        if d is not None:
                            depths.append(torch.as_tensor(d).float().flatten())
                    row["mae_" + tag] = float(torch.cat(errs).mean())
                    if depths:
                        row["depth_" + tag] = float(torch.cat(depths).mean())
            row["ratio_adaptive_over_full"] = (
                row["mae_adaptive"] / row["mae_full_depth"] if row["mae_full_depth"] else None)
            print(json.dumps(row), flush=True)
            out.append(row)
        except Exception:
            import traceback
            traceback.print_exc()
    json.dump(out, open("/root/fast/exec_mae.json", "w"), indent=2)
    print("DONE %d" % len(out), flush=True)


if __name__ == "__main__":
    main()
