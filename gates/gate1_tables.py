#!/usr/bin/env python3
"""Render the Gate-1 follow-up tables from the JSON deliverables.

Reads gates/replay_verdicts.json + gates/gate1_frequency_sweep.json and prints
markdown tables plus the cross-rate failure-set analysis.  No simulation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

G = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/zoomq/gates")
V = json.load(open(G / "replay_verdicts.json"))
S = json.load(open(G / "gate1_frequency_sweep.json"))
TASKS = [t for t in ("move_plate", "drawer_top_close") if t in S]


def se(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


print("## Task 1 -- per-demo replay verdicts (floating era, configured rate)\n")
print("| task | dsr | Hz | ms/step | n | replay success | drop fraction |")
print("|---|---|---|---|---|---|---|")
for t in TASKS:
    e = V[t]
    print(f"| `{t}` | {e['demo_down_sample_rate']} | {e['control_frequency']} | "
          f"{e['control_step_seconds'] * 1000:.0f} | {e['n']} | "
          f"{e['n_success']}/{e['n']} = {e['success_rate']:.3f} | "
          f"**{e['drop_fraction']:.3f}** |")

print("\n## Task 2.4 -- replay success vs control rate\n")
for t in TASKS:
    pr = S[t]["per_rate"]
    rates = sorted(pr, key=lambda k: int(k))
    n = pr[rates[0]]["n"]
    print(f"\n### `{t}` (n = {n} demos per cell)\n")
    print("| dsr | control Hz | ms / control step | substeps | replay success | "
          "n_success | +-1 se | demo's own recorded success |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rates:
        x = pr[r]
        star = " **(configured)**" if int(r) == 10 else ""
        print(f"| {x['demo_down_sample_rate']}{star} | {x['control_frequency_hz']} | "
              f"{x['control_step_seconds'] * 1000:.0f} | "
              f"{x['sub_steps_per_control_step']} | "
              f"**{x['success_rate']:.3f}** | {x['n_success']}/{x['n']} | "
              f"{se(x['success_rate'], x['n']):.3f} | "
              f"{x['demo_recorded_success_rate']:.3f} |")
    cr = S[t]["cross_rate"]
    print(f"\nCross-rate failure sets: {cr['n_fail_at_no_rate']} demos never fail, "
          f"{cr['n_fail_at_every_rate']} fail at EVERY rate, "
          f"{cr['n_fail_at_some_rate']} fail at >=1 rate "
          f"(of {cr['n_demos']}). Mean pairwise Jaccard of the failing sets = "
          f"{cr['mean_jaccard']:.3f}.")
    print(f"Histogram, #demos by #rates-they-fail-at: {cr['histogram_n_rates_failed']}")
    # neighbour-pair Jaccard (adjacent rates) vs far pairs
    j = cr["jaccard_of_failure_sets"]
    print(f"Jaccard vs the configured rate 10: "
          + ", ".join(f"{k.replace('_vs_', ' vs ')}={v:.2f}"
                      for k, v in sorted(j.items(), key=lambda kv: int(kv[0].split('_')[0]))
                      if k.endswith("_vs_10") or k.startswith("10_vs_")))

print("\n## Task 2.5 -- do the SAME demos fail at every rate?\n")
for t in TASKS:
    pr = S[t]["per_rate"]
    rates = sorted(pr, key=lambda k: int(k))
    fails = {r: set(pr[r]["fail_uuids"]) for r in rates}
    ever = sorted(set().union(*fails.values()))
    if not ever:
        print(f"`{t}`: no demo fails at any rate.\n")
        continue
    idx = {d["uuid"]: d["index"] for d in pr["10"]["per_demo"]}
    rec = {d["uuid"]: pr["10"]["per_demo"] for d in pr["10"]["per_demo"]}
    print(f"\n### `{t}` -- every demo that fails at >=1 rate "
          f"({len(ever)} of {pr['10']['n']})\n")
    print("| index | uuid | " + " | ".join(f"dsr{r}" for r in rates) + " | #fail |")
    print("|---" * (len(rates) + 3) + "|")
    for u in sorted(ever, key=lambda u: idx[u]):
        marks = ["FAIL" if u in fails[r] else "ok" for r in rates]
        print(f"| {idx[u]} | `{u[:8]}` | " + " | ".join(marks) + " | "
              + str(sum(m == 'FAIL' for m in marks)) + " |")

print("\n## Task 1 -- systematic or random?\n")
for t in TASKS:
    fa = V[t]["failure_analysis"]
    print(f"\n### `{t}`\n")
    if V[t]["n_success"] == V[t]["n"]:
        print("No failures.\n")
        continue
    print("| covariate | mean over replay-successes | mean over replay-failures | "
          "point-biserial r with success |")
    print("|---|---|---|---|")
    for k in ("demo_steps", "demo_first_reward_step", "demo_recorded_reward",
              "clipped_frac"):
        print(f"| `{k}` | {fa[f'mean_{k}_success']:.3f} | "
              f"{fa[f'mean_{k}_fail']:.3f} | {fa[f'corr_success_vs_{k}']:+.3f} |")
    print(f"\nInitial condition: only {fa['varying_init_qpos_coords']} of the "
          f"{len(V[t]['per_demo'][0]) and ''}scene qpos coordinates vary across the "
          f"reset seeds; the strongest |corr| between any of them and replay "
          f"success is {fa['max_abs_corr_init_qpos_vs_success']:.3f}.")
    for c in fa["top_init_qpos_correlates"][:3]:
        print(f"  - qpos[{c['qpos_index']}]: std {c['std_all']:.4f}, "
              f"mean(success) {c['mean_success']:.4f}, "
              f"mean(fail) {c['mean_fail']:.4f}, |r| {c['abs_corr_with_success']:.3f}")
