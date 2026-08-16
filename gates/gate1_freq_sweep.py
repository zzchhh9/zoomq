#!/usr/bin/env python3
"""Task 2 -- is the raw-replay ceiling a physics-version artefact or a CONTROL
RATE artefact?

Gate 1 replayed the stock demos at the configured ``demo_down_sample_rate=10``
(50 Hz) and got 0.750 on ``move_plate``, and attributed the missing 0.250 to
MuJoCo drift (demos recorded under 3.1.5, replayed under 3.11.0).  The
alternative it never tested is a control-rate mismatch, the failure mode
documented in ``third_party/CQN-AS-G1/CLAUDE.md`` ("G1 control rate era
boundary").  This sweeps the rate.

There is exactly ONE rate knob.  ``bigym.loco.env`` builds the env with
``control_frequency = CONTROL_FREQUENCY_MAX // demo_down_sample_rate`` and
``DemoStore`` is asked for demos decimated to the SAME frequency, so the demo
rate and the control rate cannot be set independently: sweeping ``dsr`` sweeps
both together, which is exactly the "was the demo recorded at the rate we
replay it at" question.  Legal range: ``500 // dsr`` must land in
[CONTROL_FREQUENCY_MIN, CONTROL_FREQUENCY_MAX] = [20, 500], i.e. dsr <= 25.

Per rate this runs the same raw open-loop replay as
``gate1_replay.py --mode raw --era floating`` and keeps the per-demo verdict,
so the failing SET can be compared across rates (same demos failing => the
demos; different demos failing => timing sensitivity).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.environ.get("GATE1_REPO", "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
GATES = str(Path(__file__).resolve().parent)
for _p in (REPO, GATES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from gate1_replay import TASK_CONFIG, run_config  # noqa: E402

DEFAULT_CACHE = f"{GATES}/bigym_cache"


def sweep_one(task, dsr, cache_root, workers, hold, keep_first=False):
    args = Namespace(
        mode="raw", era="floating", hold=hold, task=task, dsr=int(dsr),
        cache_root=cache_root, order="uuid", num_demos=TASK_CONFIG[task]["n_demos"],
        j=[16], chunk_len=16, workers=int(workers), untrimmed=True,
        era_sweep=False, seed=0, out=None, keep_first=bool(keep_first),
    )
    t0 = time.time()
    res = run_config(args, "floating", {}, hold, TASK_CONFIG[task]["n_demos"])
    rows = sorted(res["rows"], key=lambda r: r["index"])
    n = len(rows)
    fail = [r for r in rows if not r["success"]]
    return {
        "task": task,
        "keep_first": bool(keep_first),
        "demo_down_sample_rate": int(dsr),
        "control_frequency_hz": 500 // int(dsr),
        "control_step_seconds": 0.002 * int(dsr),
        "sub_steps_per_control_step": int(dsr),
        "n": n,
        "n_success": n - len(fail),
        "success_rate": (n - len(fail)) / n,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_replay_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_demo_steps": float(np.mean([r["demo_steps"] for r in rows])),
        "mean_clipped_frac": float(np.mean([r["clipped_frac"] for r in rows])),
        "demo_recorded_success_rate": float(np.mean(
            [r["demo_reward"] >= 0.25 for r in rows])),
        "fail_uuids": sorted(r["uuid"] for r in fail),
        "fail_indices": sorted(int(r["index"]) for r in fail),
        "per_demo": [{"index": int(r["index"]), "uuid": r["uuid"],
                      "seed": int(r["seed"]), "success": bool(r["success"]),
                      "reward": float(r["reward"]), "steps": int(r["steps"]),
                      "first_reward_step": int(r["first_reward_step"]),
                      "demo_steps": int(r["demo_steps"])}
                     for r in rows],
        "wall_seconds": time.time() - t0,
    }


def cross_rate(per_rate):
    """Same demos failing at every rate, or different ones?"""
    rates = sorted(per_rate, key=lambda k: int(k))
    fails = {r: set(per_rate[r]["fail_uuids"]) for r in rates}
    allu = set()
    for r in rates:
        allu |= set(d["uuid"] for d in per_rate[r]["per_demo"])
    n_all = len(allu)
    ever = set().union(*fails.values()) if fails else set()
    always = set.intersection(*fails.values()) if fails else set()
    jac = {}
    for i, a in enumerate(rates):
        for b in rates[i + 1:]:
            u = fails[a] | fails[b]
            jac[f"{a}_vs_{b}"] = (len(fails[a] & fails[b]) / len(u)) if u else 1.0
    n_fail_by_uuid = {u: sum(u in fails[r] for r in rates) for u in ever}
    return {
        "rates": [int(r) for r in rates],
        "n_demos": n_all,
        "n_fail_at_every_rate": len(always),
        "n_fail_at_some_rate": len(ever),
        "n_fail_at_no_rate": n_all - len(ever),
        "fail_at_every_rate_uuids": sorted(always),
        "jaccard_of_failure_sets": jac,
        "mean_jaccard": float(np.mean(list(jac.values()))) if jac else float("nan"),
        "histogram_n_rates_failed": {
            str(k): int(sum(1 for v in n_fail_by_uuid.values() if v == k))
            for k in range(1, len(rates) + 1)},
        "note": ("Jaccard ~1 => the SAME demos fail at every rate (intrinsic to "
                 "those demos). Jaccard ~ chance => the failing set moves with "
                 "the rate (a timing-sensitivity story)."),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="move_plate,drawer_top_close")
    p.add_argument("--rates", default="5,8,10,12,15,20,25",
                   help="demo_down_sample_rate values (<=25: 500//dsr must be "
                        ">= CONTROL_FREQUENCY_MIN=20)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--cache-root", default=DEFAULT_CACHE)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--keep-first", action="store_true", default=False,
                   help="execute the demo action[0] as well (see gate1_replay.py)")
    p.add_argument("--out", default=f"{GATES}/gate1_frequency_sweep.json")
    args = p.parse_args()

    rates = [int(x) for x in args.rates.split(",") if x.strip()]
    for d in rates:
        if not (20 <= 500 // d <= 500):
            sys.exit(f"dsr={d} -> {500 // d} Hz is outside bigym's legal "
                     "[20, 500] control-frequency range")
    import mujoco
    out = {"_meta": {
        "era": "floating", "success_hold_seconds": args.hold,
        "untrimmed": True, "mujoco_version_replay": mujoco.__version__,
        "CONTROL_FREQUENCY_MAX": 500, "CONTROL_FREQUENCY_MIN": 20,
        "configured_dsr": 10,
        "rate_knob_note": (
            "control_frequency = 500 // demo_down_sample_rate, and DemoStore is "
            "asked for demos decimated to the same frequency "
            "(bigym/loco/env.py BiGym.get_demos). The two cannot be set apart."),
        "cache_root": args.cache_root,
        "success_predicate": "episode_reward >= 0.25",
        "keep_first": None,
    }}
    out["_meta"]["keep_first"] = bool(args.keep_first)
    t0 = time.time()
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        per_rate = {}
        for d in rates:
            r = sweep_one(task, d, args.cache_root, args.workers, args.hold,
                          keep_first=args.keep_first)
            per_rate[str(d)] = r
            print(f"[{task}] dsr={d:2d} ({r['control_frequency_hz']:3d} Hz, "
                  f"{r['control_step_seconds'] * 1000:.1f} ms/step) "
                  f"success={r['success_rate']:.3f} "
                  f"({r['n_success']}/{r['n']})  "
                  f"clip={r['mean_clipped_frac']:.3f}  "
                  f"[{r['wall_seconds']:.0f}s]", flush=True)
        out[task] = {"per_rate": per_rate, "cross_rate": cross_rate(per_rate)}
    out["_wall_seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
