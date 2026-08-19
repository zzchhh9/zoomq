#!/usr/bin/env python3
"""One-snapshot exec-MAE tick. Exit 0 if wrote a row, 2 if nothing new, 1 on fail.

Same weights: descend(adaptive=True) vs descend(adaptive=False).
CPU. Do not import zoomq at module scope -- pickle load pulls it in.
"""
from __future__ import annotations

import faulthandler
import glob
import json
import os
import sys
import time

faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch

RUNS = "/mnt/workspace/zoomq/runs"
LOG = "/mnt/workspace/zoomq/logs/p13_exec_mae.jsonl"
SEEN = "/mnt/workspace/zoomq/logs/p13_exec_mae.seen"
N, T, D = 128, 16, 15
WATCH = ["p13_mp_zqEAd_s1", "p13_mp_zqEAd_s2"]
REFS = [
    ("p9_mp_zqEA_s1", "eval_snapshots/snapshot_2100.pt"),
    ("p9_mp_zqEA_s1", "eval_snapshots/snapshot_10158.pt"),
    ("p9_mp_zqEA_s1", "snapshot.pt"),
]


def load_seen():
    if not os.path.exists(SEEN):
        return set()
    return {ln.strip() for ln in open(SEEN) if ln.strip()}


def mark_seen(key):
    os.makedirs(os.path.dirname(SEEN), exist_ok=True)
    with open(SEEN, "a") as f:
        f.write(key + "\n")


def csv_frame(arm):
    p = os.path.join(RUNS, arm, "train.csv")
    if not os.path.exists(p):
        return 0
    import csv

    fr = 0
    for r in csv.DictReader(open(p)):
        if r.get("frame"):
            try:
                fr = max(fr, int(float(r["frame"])))
            except ValueError:
                pass
    return fr


def next_job(seen):
    for arm in WATCH:
        ev = os.path.join(RUNS, arm, "eval_snapshots")
        if os.path.isdir(ev):
            for p in sorted(glob.glob(os.path.join(ev, "snapshot_*.pt"))):
                key = arm + ":" + os.path.basename(p)
                if key not in seen:
                    return key, arm, p, "eval"
        live = os.path.join(RUNS, arm, "snapshot.pt")
        if os.path.exists(live):
            fr = csv_frame(arm)
            bucket = (fr // 300) * 300
            key = "%s:live:%d" % (arm, bucket)
            if key not in seen and fr > 0:
                return key, arm, live, "live:%d" % fr
    for arm, rel in REFS:
        p = os.path.join(RUNS, arm, rel)
        key = arm + ":" + rel
        if os.path.exists(p) and key not in seen:
            return key, arm, p, "ref"
    return None


@torch.no_grad()
def pair_mae(cr, feat, low, demo):
    mr = int(getattr(cr, "num_rounds", 5)) - 1

    def one(adaptive):
        acts, depths = [], []
        for i in range(0, feat.shape[0], 16):
            a, info = cr.descend(
                feat[i : i + 16],
                low[i : i + 16],
                max_round=mr,
                adaptive=adaptive,
                kappa=1.0,
                forced_depth=None,
                early_exit=bool(adaptive),
            )
            acts.append(a.reshape(-1, T, D))
            depths.append(info["depth"])
        ah = torch.cat(acts)
        dep = torch.cat(depths)
        hist = {int(k): int((dep == k).sum()) for k in range(mr + 1)}
        mae = float((ah - demo).abs().mean())
        modal = int(max(hist, key=hist.get)) if hist else -1
        return mae, hist, modal

    mae_ad, hist_ad, mod_ad = one(True)
    mae_fu, hist_fu, mod_fu = one(False)
    ratio = (mae_ad / mae_fu) if mae_fu > 1e-12 else None
    return {
        "mae_adaptive": mae_ad,
        "mae_full_depth": mae_fu,
        "ratio_ad_over_full": None if ratio is None else round(ratio, 4),
        "depth_hist_adaptive": hist_ad,
        "depth_hist_full": hist_fu,
        "modal_adaptive": mod_ad,
        "modal_full": mod_fu,
    }


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.manual_seed(0)
    sys.path.insert(0, "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
    sys.path.insert(0, "/root/gap")
    from critic_discrim import build_batch, load

    seen = load_seen()
    job = next_job(seen)
    if job is None:
        print("tick: nothing new", flush=True)
        return 2
    key, arm, path, tag = job
    print("tick: %s %s" % (arm, tag), flush=True)
    rgb, low, act = build_batch(N)
    rgb = torch.as_tensor(rgb).float()
    low = torch.as_tensor(low)
    act = torch.as_tensor(act)
    demo = act.view(-1, T, D)
    t0 = time.time()
    ag = load(path)
    enc = ag.encoder.float().cpu().eval()
    cr = ag.critic.float().cpu().eval()
    feat = torch.cat([enc(rgb[i : i + 8]) for i in range(0, rgb.shape[0], 8)])
    rec = pair_mae(cr, feat, low, demo)
    rec.update(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "key": key,
            "arm": arm,
            "path": path,
            "tag": tag,
            "cfg_adaptive": bool(getattr(ag, "zq_adaptive", None)),
            "train_frame": csv_frame(arm),
            "n": int(rgb.shape[0]),
            "secs": round(time.time() - t0, 1),
        }
    )
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    mark_seen(key)
    print(
        "  %-22s %-16s full=%.4f ad=%.4f ratio=%s modal_ad=%s %ss"
        % (
            arm,
            tag,
            rec["mae_full_depth"],
            rec["mae_adaptive"],
            rec["ratio_ad_over_full"],
            rec["modal_adaptive"],
            rec["secs"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("FAIL:", e, flush=True)
        sys.exit(1)
