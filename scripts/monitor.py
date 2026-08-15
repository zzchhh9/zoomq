#!/usr/bin/env python3
"""Summarise ZoomQ training arms from their train.csv / eval.csv.

Usage:
    python scripts/monitor.py [--runs /mnt/workspace/zoomq/runs] [--json OUT]

Prints one table of shared progress columns and, for arms that have them, a
second table of the ZoomQ mechanism counters. The counters are the point of the
night, so they are printed even when they are degenerate — a depth histogram
pinned at 0% or 100% is a result (of some kind), not a reason to hide the row.
"""

import argparse
import csv
import json
import math
from pathlib import Path

ZQ_KEYS = [
    "depth_share_0", "depth_share_1", "depth_share_2", "depth_share_3", "depth_share_4",
    "exec_n_0", "exec_n_1", "exec_n_2", "exec_n_3", "exec_n_4",
    "exec_q_0", "exec_q_1", "exec_q_2", "exec_q_3", "exec_q_4",
    "exec_td_err_0", "exec_td_err_1", "exec_td_err_2", "exec_td_err_3", "exec_td_err_4",
    "skeleton_rmse_0", "skeleton_rmse_1", "skeleton_rmse_2", "skeleton_rmse_3",
    "window_clamp_rate", "refine_target_mass", "bellman_target_mass",
    "refine_target_q", "bellman_target_q", "consistency_drift",
    "eps_depth_current", "delta_over_kappa_u_p10", "delta_over_kappa_u_p50",
    "delta_over_kappa_u_p90", "zoomq_exec_loss", "zoomq_refine_loss",
]


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def f(row, key, default=float("nan")):
    try:
        v = float(row[key])
        return v if math.isfinite(v) else default
    except (KeyError, TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/mnt/workspace/zoomq/runs")
    ap.add_argument("--json", default=None, help="also write the summary as JSON")
    args = ap.parse_args()

    root = Path(args.runs)
    out = {}
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        tr = read_csv(d / "train.csv")
        ev = read_csv(d / "eval.csv")
        if not tr:
            rows.append((d.name, 0, 0.0, 0.0, 0, float("nan"), float("nan")))
            out[d.name] = {"frames": 0, "note": "no train.csv yet"}
            continue
        last = tr[-1]
        frames = f(last, "frame", 0)
        total_t = f(last, "total_time", 0)
        fps = frames / total_t if total_t > 0 else float("nan")
        evals = len(ev)
        last_ev = f(ev[-1], "episode_reward") if ev else float("nan")
        best_ev = max((f(r, "episode_reward", 0.0) for r in ev), default=float("nan"))
        rows.append((d.name, int(frames), fps, total_t / 3600.0, evals, last_ev, best_ev))
        rec = {
            "frames": int(frames), "fps": round(fps, 3),
            "hours": round(total_t / 3600.0, 2), "evals": evals,
            "last_eval_reward": last_ev, "best_eval_reward": best_ev,
            "episode_length_last_eval": f(ev[-1], "episode_length") if ev else None,
        }
        zq = {k: f(last, k) for k in ZQ_KEYS if k in last}
        if zq:
            rec["zoomq"] = {k: round(v, 5) for k, v in zq.items()}
        out[d.name] = rec

    w = max((len(r[0]) for r in rows), default=4)
    print(f"{'arm':<{w}}  {'frames':>8} {'fps':>6} {'hours':>6} {'evals':>5} "
          f"{'last_R':>8} {'best_R':>8}")
    print("-" * (w + 48))
    for name, fr, fps, hrs, ne, lr, br in rows:
        print(f"{name:<{w}}  {fr:>8d} {fps:>6.2f} {hrs:>6.2f} {ne:>5d} "
              f"{lr:>8.4f} {br:>8.4f}")

    zq_arms = [(k, v["zoomq"]) for k, v in out.items() if "zoomq" in v]
    if zq_arms:
        print("\nZoomQ mechanism counters (latest train row)")
        print("-" * 78)
        for name, zq in zq_arms:
            ds = [zq.get(f"depth_share_{r}", float('nan')) for r in range(5)]
            en = [zq.get(f"exec_n_{r}", float('nan')) for r in range(5)]
            eq = [zq.get(f"exec_q_{r}", float('nan')) for r in range(5)]
            print(f"{name}:")
            print(f"   depth_share  {' '.join(f'{x:6.3f}' for x in ds)}"
                  f"   <- rollout depth histogram")
            print(f"   exec_n       {' '.join(f'{x:6.3f}' for x in en)}"
                  f"   <- fraction of batch valid per depth")
            print(f"   exec_q       {' '.join(f'{x:6.3f}' for x in eq)}")
            print(f"   clamp={zq.get('window_clamp_rate', float('nan')):.4f}  "
                  f"refine_mass={zq.get('refine_target_mass', float('nan')):.4f}  "
                  f"drift={zq.get('consistency_drift', float('nan')):.4f}  "
                  f"eps={zq.get('eps_depth_current', float('nan')):.3f}  "
                  f"d/ku p50={zq.get('delta_over_kappa_u_p50', float('nan')):.3f}")
            flags = []
            if ds[0] > 0.99:
                flags.append("depth pinned at round 0 (self-lock fixed point, "
                             "the F5 refine-backup bug, or simply no reward signal)")
            if ds[-1] > 0.99:
                flags.append("depth pinned at max round")
            rm = zq.get("refine_target_mass", float("nan"))
            bm = zq.get("bellman_target_mass", float("nan"))
            # Every refine target is either a copied exec distribution (mass
            # exactly 1) or the Bellman leaf, whose mass sits slightly above 1
            # because the baseline's C51 projection duplicates mass on
            # exact-atom hits (c51_exact_atom_fix: false, kept so arm C stays
            # comparable to arm B). So refine mass must lie between the two.
            # The F5 bug would instead give a delta at V=0 with mass ~2.
            if math.isfinite(rm) and math.isfinite(bm):
                lo, hi = min(1.0, bm) - 1e-3, max(1.0, bm) + 1e-3
                if not (lo <= rm <= hi):
                    flags.append(
                        f"refine mass {rm:.4f} outside [1.0, bellman {bm:.4f}] "
                        "-> a projection leaked into the refine backup (eq. 2)")
                elif bm > 1.0 + 1e-6:
                    # how often W_r = Q^exec_r beat the child, inferred from the
                    # mixture: exec branches contribute mass 1, the leaf mass bm
                    share = max(0.0, min(1.0, (bm - rm) / (bm - 1.0)))
                    print(f"   exec-branch share of refine targets ~ {share:.3f} "
                          "(derived from the mass mixture)")
            if zq.get("window_clamp_rate", 0.0) > 0.05:
                flags.append("clamp rate > 5% -> windows mis-specified, the run is "
                             "measuring a clipped object")
            for fl in flags:
                print(f"   !! {fl}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
