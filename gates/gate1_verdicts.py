#!/usr/bin/env python3
"""Task 1 -- per-demonstration replay verdicts, as a file a trainer can consume.

Replays every stock BiGym demonstration action-for-action on the FLOATING base
(``lowerbody_policy.enabled=false``, ``base_action_mode=legacy_delta``,
``success_hold_seconds=0``) at the CONFIGURED control rate
(``demo_down_sample_rate`` from ``cfgs/bigym_task/<task>.yaml``) and records,
per demo, whether it succeeds.

The point of the file is to let a trainer admit only demonstrations that
actually reach the goal in this environment.  That needs a STABLE KEY, and

    THE POSITION IN THE LIST ``DemoStore`` RETURNS IS NOT ONE.

``DemoStore._get_demos()`` ends in ``np.random.shuffle(files)``, so the order
``BiGym.get_demos`` -> ``train_cqn_as_bigym.py`` sees depends on the global
numpy RNG state at load time (verified: same np seed -> same order, different
np seed -> different order).  The stable keys are

    * ``uuid``  -- the demo's ``<uuid>.safetensors`` filename stem, and
    * ``seed``  -- its recorded env reset seed,

both unique across the demo set for both tasks.  ``index`` in this file is the
rank of the demo in ``sorted(uuid)`` order: deterministic, derivable from the
demo directory listing alone, and NOT the trainer's load position.  A consumer
must join on ``uuid`` (or ``seed``), never on a positional index.

Also records the reset-time ``qpos`` of the scene so the failures can be tested
for clustering in initial conditions.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.environ.get("GATE1_REPO", "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
GATES = str(Path(__file__).resolve().parent)
for _p in (REPO, GATES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from gate1_replay import TASK_CONFIG, build_env, load_demos  # noqa: E402

DEFAULT_CACHE = f"{GATES}/bigym_cache"


def _joint_names(model):
    import mujoco

    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)]


def _worker(payload):
    task, dsr, cache_root, shard, hold, keep_first = payload
    env = build_env("floating", hold, None, task=task, dsr=dsr)
    inner = env._env
    raw_env = inner._env
    model = raw_env._mojo.model
    data = raw_env._mojo.data
    demos = load_demos(env, untrimmed=True, cache_root=cache_root,
                       order="uuid", keep_first=keep_first)
    meta = {
        "control_frequency": int(raw_env.control_frequency),
        "control_step_seconds": float(raw_env.control_step_seconds),
        "sub_steps": int(raw_env._sub_steps_count),
        "compiled_timestep_s": float(model.opt.timestep),
        "joint_names": _joint_names(model),
        "nq": int(model.nq),
        "n_total_demos": len(demos),
    }
    rows = []
    for i in shard:
        d = demos[i]
        env.reset(seed=int(d["seed"]))
        init_qpos = np.array(data.qpos, dtype=np.float64).tolist()
        rew, steps, first = 0.0, 0, -1
        for a in d["actions"]:
            ts = env.step(np.asarray(a, dtype=np.float32))
            if first < 0 and float(ts.reward) > 0:
                first = steps
            rew += float(ts.reward)
            steps += 1
            if ts.last():
                break
        rows.append({
            "index": int(d["index"]),
            "uuid": d["uuid"],
            "seed": int(d["seed"]),
            "seed_rank": int(d["seed_rank"]),
            "success": bool(rew >= 0.25),
            "reward": float(rew),
            "steps": int(steps),
            "first_reward_step": int(first),
            "demo_steps": int(d["demo_steps"]),
            "demo_first_reward_step": int(d["demo_first_reward_step"]),
            "demo_recorded_reward": float(d["demo_reward"]),
            "clipped_frac": float(d["clipped_frac"]),
            "init_qpos": init_qpos,
        })
    env.close()
    return meta, rows


def run_task(task, dsr, cache_root, workers, hold=0.0, keep_first=True):
    n = TASK_CONFIG[task]["n_demos"]
    shards = [list(range(i, n, workers)) for i in range(workers)]
    shards = [s for s in shards if s]
    payloads = [(task, dsr, cache_root, s, hold, keep_first) for s in shards]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        parts = pool.map(_worker, payloads)
    meta = parts[0][0]
    rows = sorted((r for _, rs in parts for r in rs), key=lambda r: r["index"])
    assert len(rows) == n, f"{task}: got {len(rows)} rows, expected {n}"
    assert len({r["uuid"] for r in rows}) == n, f"{task}: duplicate uuids"
    return meta, rows


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyse(rows, meta):
    """Are the failures systematic (initial condition / length) or random?"""
    succ = np.array([r["success"] for r in rows], dtype=float)
    out = {}
    for key in ("demo_steps", "demo_first_reward_step", "clipped_frac",
                "demo_recorded_reward"):
        v = [r[key] for r in rows]
        out[f"corr_success_vs_{key}"] = _corr(succ, v)
        out[f"mean_{key}_success"] = float(np.mean(
            [x for x, s in zip(v, succ) if s])) if succ.any() else float("nan")
        out[f"mean_{key}_fail"] = float(np.mean(
            [x for x, s in zip(v, succ) if not s])) if (1 - succ).any() \
            else float("nan")
    # initial condition: every qpos coordinate, success vs fail
    q = np.array([r["init_qpos"] for r in rows], dtype=float)
    names = meta["joint_names"]
    per_coord = []
    for k in range(q.shape[1]):
        col = q[:, k]
        if col.std() < 1e-12:
            continue
        per_coord.append({
            "qpos_index": k,
            "mean_success": float(col[succ > 0].mean()) if succ.any() else None,
            "mean_fail": float(col[succ == 0].mean()) if (1 - succ).any() else None,
            "std_all": float(col.std()),
            "abs_corr_with_success": abs(_corr(succ, col)),
        })
    per_coord.sort(key=lambda r: -r["abs_corr_with_success"])
    out["varying_init_qpos_coords"] = len(per_coord)
    out["init_qpos_joint_names"] = names
    out["top_init_qpos_correlates"] = per_coord[:8]
    out["max_abs_corr_init_qpos_vs_success"] = (
        per_coord[0]["abs_corr_with_success"] if per_coord else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="move_plate,drawer_top_close")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--cache-root", default=DEFAULT_CACHE)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--out", default=f"{GATES}/replay_verdicts.json")
    args = p.parse_args()

    t0 = time.time()
    result = {
        "_schema": {
            "key": "uuid",
            "index_semantics": (
                "index = rank of the demo in sorted(uuid) order. It is NOT the "
                "position DemoStore/BiGym.get_demos returns: "
                "DemoStore._get_demos() ends in np.random.shuffle(files), so the "
                "trainer's load order depends on the global numpy RNG state. "
                "JOIN ON uuid (or seed), never on a positional index."),
            "success_predicate": "episode_reward >= 0.25 (the repo's eval predicate)",
            "era": "floating (lowerbody_policy.enabled=false, "
                   "base_action_mode=legacy_delta, support_base_with_legs=false)",
            "untrimmed": True,
            "actions_executed": ("all recorded demo actions. The cached "
                "DemoStore demos are built by "
                "DemoConverter.create_demo_in_new_env, which env.reset()s and "
                "then steps with EVERY source action, so timestep 0 is "
                "already a post-action state -- there is no dummy reset "
                "action to skip. gates/gate1_report.md skipped it; the cost "
                "on move_plate at 50 Hz is 0.867 -> 0.750. Both are "
                "reported."),
            "mujoco_version_replay": None,
        },
    }
    import mujoco
    result["_schema"]["mujoco_version_replay"] = mujoco.__version__

    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        print(f"[{task}] dsr={TASK_CONFIG[task].get('dsr', 10)}", flush=True)
        dsr = 10
        meta, rows = run_task(task, dsr, args.cache_root, args.workers,
                              args.hold, keep_first=True)
        _, rows_df = run_task(task, dsr, args.cache_root, args.workers,
                              args.hold, keep_first=False)
        df = {r["uuid"]: r for r in rows_df}
        for r in rows:
            d = df[r["uuid"]]
            r["success_gate1_dropfirst"] = bool(d["success"])
            r["reward_gate1_dropfirst"] = float(d["reward"])
        n_succ = int(sum(r["success"] for r in rows))
        n_succ_df = int(sum(r["success_gate1_dropfirst"] for r in rows))
        entry = {
            "era": "floating",
            "success_hold_seconds": float(args.hold),
            "control_frequency": meta["control_frequency"],
            "control_step_seconds": meta["control_step_seconds"],
            "sub_steps_per_control_step": meta["sub_steps"],
            "compiled_mujoco_timestep_s": meta["compiled_timestep_s"],
            "demo_down_sample_rate": dsr,
            "enable_all_floating_dof": TASK_CONFIG[task]["enable_all_floating_dof"],
            "episode_length": TASK_CONFIG[task]["episode_length"],
            "n": len(rows),
            "n_success": n_succ,
            "success_rate": n_succ / len(rows),
            "drop_fraction": 1.0 - n_succ / len(rows),
            "success_indices": [r["index"] for r in rows if r["success"]],
            "fail_indices": [r["index"] for r in rows if not r["success"]],
            "success_uuids": [r["uuid"] for r in rows if r["success"]],
            "fail_uuids": [r["uuid"] for r in rows if not r["success"]],
            "success_seeds": [r["seed"] for r in rows if r["success"]],
            "fail_seeds": [r["seed"] for r in rows if not r["success"]],
            "actions_executed": "all",
            "gate1_dropfirst_convention": {
                "actions_executed": "all-but-index-0",
                "n_success": n_succ_df,
                "success_rate": n_succ_df / len(rows),
                "drop_fraction": 1.0 - n_succ_df / len(rows),
                "fail_indices": [r["index"] for r in rows
                                 if not r["success_gate1_dropfirst"]],
                "fail_uuids": [r["uuid"] for r in rows
                               if not r["success_gate1_dropfirst"]],
                "note": ("the convention gates/gate1_report.md published "
                         "(0.750 on move_plate). It skips the demo's first "
                         "recorded action, which the cached DemoStore demos "
                         "do NOT carry as a dummy."),
            },
            "failure_analysis": analyse(rows, meta),
            "per_demo": [{k: v for k, v in r.items() if k != "init_qpos"}
                         for r in rows],
        }
        result[task] = entry
        print(f"  n={entry['n']} success={n_succ} rate={entry['success_rate']:.3f} "
              f"drop={entry['drop_fraction']:.3f}  "
              f"[gate1 drop-first convention: {n_succ_df}/{len(rows)} = "
              f"{n_succ_df / len(rows):.3f}]", flush=True)

    result["_wall_seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
