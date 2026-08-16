#!/usr/bin/env python3
"""Merge the two rate sweeps into the single gate1_frequency_sweep.json
deliverable, and add the chance baseline for the failure-set Jaccard.

PRIMARY block  = keep_first (every recorded action executed).
SECONDARY block = the historical Gate-1 convention that drops action[0].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

G = Path("/mnt/workspace/zoomq/gates")
DROP = json.load(open(G / "gate1_raw" / "sweep_dropfirst.json"))
KEEP = json.load(open(G / "gate1_raw" / "sweep_keepfirst.json"))
TASKS = [t for t in KEEP if not t.startswith("_")]


def chance_jaccard(cross, per_rate):
    """Expected Jaccard of two independent failure sets of the observed sizes."""
    out = {}
    for k in cross["jaccard_of_failure_sets"]:
        a, b = k.split("_vs_")
        na = len(per_rate[a]["fail_uuids"])
        nb = len(per_rate[b]["fail_uuids"])
        n = per_rate[a]["n"]
        inter = na * nb / n
        union = na + nb - inter
        out[k] = float(inter / union) if union > 0 else 1.0
    return out


def block(src):
    out = {}
    for t in TASKS:
        pr = src[t]["per_rate"]
        cr = dict(src[t]["cross_rate"])
        cj = chance_jaccard(cr, pr)
        cr["chance_jaccard_if_independent"] = cj
        cr["mean_chance_jaccard"] = float(np.mean(list(cj.values()))) if cj else 1.0
        rates = sorted(pr, key=lambda k: int(k))
        sr = [pr[r]["success_rate"] for r in rates]
        cr["success_rate_min"] = float(min(sr))
        cr["success_rate_max"] = float(max(sr))
        cr["success_rate_span"] = float(max(sr) - min(sr))
        cr["success_rate_at_configured_dsr10"] = float(pr["10"]["success_rate"])
        out[t] = {"per_rate": pr, "cross_rate": cr}
    return out


keep, drop = block(KEEP), block(DROP)
merged = {
    "_meta": {
        **KEEP["_meta"],
        "primary_convention": "keep_first (every recorded demo action executed)",
        "secondary_convention": (
            "gate1_dropfirst -- gates/gate1_report.md's convention, which skips "
            "the demo's action[0]. The cached DemoStore demos carry no dummy "
            "reset action (DemoConverter.create_demo_in_new_env resets then "
            "steps with every source action), so skipping it replays one action "
            "short, and the damage scales with the length of one control step."),
        "raw_files": {
            "keep_first": "gate1_raw/sweep_keepfirst.json",
            "gate1_dropfirst": "gate1_raw/sweep_dropfirst.json",
        },
    }
}
for t in TASKS:
    kp, dp = keep[t]["per_rate"], drop[t]["per_rate"]
    rates = sorted(kp, key=lambda k: int(k))
    merged[t] = {
        "per_rate": kp,
        "cross_rate": keep[t]["cross_rate"],
        "gate1_dropfirst": {"per_rate": dp, "cross_rate": drop[t]["cross_rate"]},
        "rate_dependence": {
            "keep_first_success_by_dsr": {r: kp[r]["success_rate"] for r in rates},
            "dropfirst_success_by_dsr": {r: dp[r]["success_rate"] for r in rates},
            "keep_first_span": keep[t]["cross_rate"]["success_rate_span"],
            "dropfirst_span": drop[t]["cross_rate"]["success_rate_span"],
            "verdict": None,
        },
    }
    ks = keep[t]["cross_rate"]["success_rate_span"]
    ds = drop[t]["cross_rate"]["success_rate_span"]
    n = kp["10"]["n"]
    merged[t]["rate_dependence"]["verdict"] = (
        f"span over 20-500 Hz: {ks:.3f} with every action executed vs "
        f"{ds:.3f} with Gate-1's dropped first action (n={n}, 1 se ~ "
        f"{np.sqrt(0.85 * 0.15 / n):.3f}). "
        + ("No rate dependence survives once the dropped action is restored."
           if ks <= 2 * np.sqrt(0.85 * 0.15 / n) else
           "A rate dependence survives the fix."))

out = G / "gate1_frequency_sweep.json"
with open(out, "w") as f:
    json.dump(merged, f, indent=2, default=float)
print("wrote", out)
for t in TASKS:
    rd = merged[t]["rate_dependence"]
    print(f"  {t}: {rd['verdict']}")
