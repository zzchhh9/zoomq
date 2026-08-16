#!/usr/bin/env python3
"""Materialise the DemoStore cache for one (task, demo_down_sample_rate).

``DemoStore.get_demos`` regenerates a demo set for an uncached frequency by
decimating the shipped 500 Hz originals and replaying them in a fresh env
(``DemoConverter.decimate`` + ``create_demo_in_new_env``).  That WRITES, so it
has to happen once, serially, into a private cache root before the parallel
replay fans out -- otherwise 24 workers race on the same directory.

Usage:  gate1_prebuild_rates.py <task> <dsr> <cache_root>
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.environ.get("GATE1_REPO", "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
GATES = "/mnt/workspace/zoomq/gates"
for p in (REPO, GATES):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from gate1_replay import build_env, load_demos  # noqa: E402


def main():
    task, dsr, cache_root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    env = build_env("floating", 0.0, None, task=task, dsr=dsr)
    demos = load_demos(env, untrimmed=True, cache_root=cache_root, order="uuid")
    lens = [len(d["actions"]) for d in demos]
    out = {
        "task": task, "dsr": dsr, "cache_root": cache_root,
        "control_frequency_hz": int(env._env._env.control_frequency),
        "control_step_seconds": float(env._env._env.control_step_seconds),
        "sub_steps": int(env._env._env._sub_steps_count),
        "compiled_timestep_s": float(env._env._env._mojo.model.opt.timestep),
        "n_demos": len(demos),
        "action_dim": int(demos[0]["actions"].shape[1]),
        "demo_len_min": int(min(lens)), "demo_len_max": int(max(lens)),
        "demo_len_mean": float(np.mean(lens)),
        "mean_clipped_frac": float(np.mean([d["clipped_frac"] for d in demos])),
        "demo_recorded_reward_mean": float(
            np.mean([d["demo_reward"] for d in demos])),
        "demo_recorded_success_frac": float(
            np.mean([d["demo_reward"] >= 0.25 for d in demos])),
        # fingerprint of the executed action tensor -- lets us prove whether two
        # cache builds produced identical demos
        "action_checksum": float(sum(float(np.abs(d["actions"]).sum())
                                     for d in demos)),
        "per_demo_action_sum": {d["uuid"]: float(np.abs(d["actions"]).sum())
                                for d in demos},
        "wall_seconds": time.time() - t0,
    }
    env.close()
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
