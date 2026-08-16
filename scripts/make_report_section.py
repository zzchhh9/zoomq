#!/usr/bin/env python3
"""Emit the report's "Live numbers" section straight from the run logs.

Run on ali:
    python scripts/make_report_section.py > /tmp/section7.md

Everything it prints is read from `runs/*/train.csv`, `runs/*/eval.csv` and
`git`, so no number in the report has to be typed by hand. The depth-histogram
history comes from `logs/status_series.log`, because a single end-of-night
reading cannot show whether the distribution MOVED — which is the actual
question the night was set up to answer.
"""

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path

BASELINE = "fba52f7b38fed6e12ee3983ae456a1f0e657bf6f"


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"(failed: {e})"


def rows(p):
    return list(csv.DictReader(p.open())) if p.exists() else []


def fl(row, key, default=float("nan")):
    try:
        v = float(row[key])
        return v if math.isfinite(v) else default
    except (KeyError, TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/workspace/zoomq")
    args = ap.parse_args()
    root = Path(args.root)
    sub = root / "third_party" / "CQN-AS-G1"

    head = sh(f"git -C {sub} rev-parse HEAD")
    branch = sh(f"git -C {sub} rev-parse --abbrev-ref HEAD")
    stat = sh(f"git -C {sub} diff --stat {BASELINE}..{head}")
    ncommit = sh(f"git -C {sub} rev-list --count {BASELINE}..{head}")

    print("## 7. Live numbers\n")
    print(f"Generated {sh('date -Iseconds')} by `scripts/make_report_section.py`; "
          "every figure below is read from the run logs.\n")
    print("### 7.1 Code\n")
    print(f"- submodule `third_party/CQN-AS-G1` @ **`{head[:12]}`** on branch `{branch}` "
          f"({ncommit} commits on top of the baseline)")
    print(f"- baseline SHA `{BASELINE[:12]}`\n")
    print("```")
    print(f"git diff {BASELINE[:12]}..zoomq --stat")
    print(stat)
    print("```\n")

    print("### 7.2 Per-arm progress\n")
    print("| arm | frames | frames/s | hours | evals | last eval R | best eval R |")
    print("|---|---|---|---|---|---|---|")
    zq = {}
    for d in sorted((root / "runs").iterdir()):
        if not d.is_dir():
            continue
        tr, ev = rows(d / "train.csv"), rows(d / "eval.csv")
        if not tr:
            print(f"| {d.name} | 0 | — | — | 0 | — | — |")
            continue
        last = tr[-1]
        fr, t = fl(last, "frame", 0), fl(last, "total_time", 0)
        fps = fr / t if t else float("nan")
        le = fl(ev[-1], "episode_reward") if ev else float("nan")
        be = max((fl(r, "episode_reward", 0.0) for r in ev), default=float("nan"))
        print(f"| {d.name} | {int(fr)} | {fps:.2f} | {t/3600:.2f} | {len(ev)} | "
              f"{'—' if math.isnan(le) else f'{le:.4f}'} | "
              f"{'—' if math.isnan(be) else f'{be:.4f}'} |")
        if "depth_share_0" in last:
            zq[d.name] = (tr[0], last)

    print("\n### 7.3 ZoomQ mechanism counters — first logged row vs last\n")
    print("`depth_share` is the rollout depth histogram (training rollouts only); "
          "`exec_n` is the fraction of the batch that is a valid label for each "
          "depth's execution head.\n")
    for name, (first, last) in zq.items():
        print(f"**{name}**\n")
        print("| counter | first | last |")
        print("|---|---|---|")
        ds_f = " / ".join(f"{fl(first, f'depth_share_{r}'):.3f}" for r in range(5))
        ds_l = " / ".join(f"{fl(last, f'depth_share_{r}'):.3f}" for r in range(5))
        en_f = " / ".join(f"{fl(first, f'exec_n_{r}'):.3f}" for r in range(5))
        en_l = " / ".join(f"{fl(last, f'exec_n_{r}'):.3f}" for r in range(5))
        eq_f = " / ".join(f"{fl(first, f'exec_q_{r}'):.3f}" for r in range(5))
        eq_l = " / ".join(f"{fl(last, f'exec_q_{r}'):.3f}" for r in range(5))
        print(f"| depth_share 0..4 | {ds_f} | {ds_l} |")
        print(f"| exec_n 0..4 | {en_f} | {en_l} |")
        print(f"| exec_q 0..4 | {eq_f} | {eq_l} |")
        for k in ("window_clamp_rate", "refine_target_mass", "bellman_target_mass",
                  "consistency_drift", "eps_depth_current",
                  "delta_over_kappa_u_p50", "zoomq_exec_loss", "zoomq_refine_loss"):
            if k in last:
                print(f"| {k} | {fl(first, k):.4f} | {fl(last, k):.4f} |")
        print()

    series = root / "logs" / "status_series.log"
    if series.exists():
        stamps = re.findall(r"^===== (\S+) =====$", series.read_text(), re.M)
        print(f"### 7.4 Time series\n")
        print(f"`logs/status_series.log` holds {len(stamps)} snapshots "
              f"({stamps[0] if stamps else '—'} to {stamps[-1] if stamps else '—'}), "
              "10 minutes apart — use it to see whether a counter moved rather "
              "than where it ended up.\n")


if __name__ == "__main__":
    main()
