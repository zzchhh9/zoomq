#!/usr/bin/env python3
"""Why does the harness replay (0.750) disagree with the demo's OWN recorded
outcome at the same rate (0.850)?

Both are "execute the demo's actions on a floating base at 50 Hz".  They differ
in three places, and this probe separates them:

  A. ``drop_first``  -- the harness executes ``actions[1:]`` (index 0 is treated
     as the reset step's dummy action); ``DemoConverter.create_demo_in_new_env``
     -- which is what produced the reward stored in the demo file -- executes
     ALL timesteps including index 0.
  B. ``normalise``   -- the harness pushes actions through the training
     normalisation (``extract_action_stats`` -> ``_convert_action_from_raw`` ->
     clip to [-1, 1]); the cache build executes the raw float64 action.  This
     arm CANNOT be run through the loco wrapper: bigym/loco/env.py
     _convert_action_to_raw asserts the incoming action is already in [-1, 1],
     so the loco env only ever sees normalised actions.  The raw-action number
     is the demo file's own stored reward (gate1_raw/prebuild/*.json).
  C. ``time_limit``  -- the loco wrapper truncates at
     ``episode_length // demo_down_sample_rate`` outer steps; the bare env used
     by the cache build has no time limit.

Everything runs in the loco env used by Gate 1, floating era, hold=0.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.environ.get("GATE1_REPO", "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
GATES = str(Path(__file__).resolve().parent)
for _p in (REPO, GATES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from gate1_replay import TASK_CONFIG, build_env  # noqa: E402

CACHE = f"{GATES}/bigym_cache"
DSR = 10


def load_variants(env, cache_root):
    """Return, per demo, both the normalised-clipped and the RAW action stream."""
    from demonstrations.demo_store import DemoStore
    from bigym.bigym_env import CONTROL_FREQUENCY_MAX

    inner = env._env
    dsr = int(inner._demo_down_sample_rate)
    store = DemoStore(Path(cache_root))
    raw = store.get_demos(inner._demo_metadata(), amount=-1,
                          frequency=CONTROL_FREQUENCY_MAX // dsr)
    for d in raw:
        for ts in d.timesteps:
            ts.observation = {k: np.array(v, dtype=np.float32)
                              for k, v in ts.observation.items()}
    raw.sort(key=lambda d: d.metadata.uuid)

    def trim(steps):
        out = []
        for ds in steps:
            out.append(ds)
            if ds.reward > 0:
                break
        return out

    conv = [inner.convert_demo_to_timesteps(trim(d.timesteps))[0] for d in raw]
    inner._action_stats = inner.extract_action_stats(conv)

    out = []
    for i, d in enumerate(raw):
        acts_raw = np.stack([ds.info["demo_action"].astype(np.float64)
                             for ds in d.timesteps])
        acts_norm = np.clip(
            inner._convert_action_from_raw(acts_raw.astype(np.float32)), -1, 1)
        out.append({"index": i, "uuid": d.metadata.uuid, "seed": int(d.seed),
                    "raw": acts_raw, "norm": acts_norm,
                    "demo_reward": float(sum(s.reward for s in d.timesteps))})
    return out


def _worker(payload):
    task, shard, cache_root = payload
    env = build_env("floating", 0.0, None, task=task, dsr=DSR)
    inner = env._env
    demos = load_variants(env, cache_root)
    rows = []
    for i in shard:
        d = demos[i]
        row = {"index": d["index"], "uuid": d["uuid"], "seed": d["seed"],
               "demo_recorded_reward": d["demo_reward"]}
        for space in ("norm",):
            for drop_first in (True, False):
                for no_limit in (False, True):
                    acts = d[space][1:] if drop_first else d[space]
                    if no_limit:
                        inner._episode_length = 10 ** 9
                    else:
                        inner._episode_length = TASK_CONFIG[task]["episode_length"]
                    env.reset(seed=int(d["seed"]))
                    rew, steps = 0.0, 0
                    for a in acts:
                        ts = env.step(np.asarray(a, dtype=np.float32))
                        rew += float(ts.reward)
                        steps += 1
                        if ts.last():
                            break
                    tag = (f"{space}_"
                           f"{'drop1' if drop_first else 'all'}_"
                           f"{'nolimit' if no_limit else 'limit'}")
                    row[tag] = {"reward": rew, "steps": steps,
                                "success": bool(rew >= 0.25)}
        inner._episode_length = TASK_CONFIG[task]["episode_length"]
        rows.append(row)
    env.close()
    return rows


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {"_meta": {"dsr": DSR, "era": "floating", "hold": 0.0,
                     "cache_root": CACHE}}
    for task in ("move_plate", "drawer_top_close"):
        n = TASK_CONFIG[task]["n_demos"]
        shards = [list(range(i, n, workers)) for i in range(workers)]
        shards = [s for s in shards if s]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(shards)) as pool:
            parts = pool.map(_worker, [(task, s, CACHE) for s in shards])
        rows = sorted((r for p in parts for r in p), key=lambda r: r["index"])
        tags = [k for k in rows[0] if isinstance(rows[0][k], dict)]
        summary = {t: float(np.mean([r[t]["success"] for r in rows])) for t in tags}
        summary["demo_recorded_success"] = float(np.mean(
            [r["demo_recorded_reward"] >= 0.25 for r in rows]))
        out[task] = {"n": len(rows), "summary": summary, "rows": rows}
        print(f"[{task}] n={len(rows)}")
        for k in sorted(summary):
            print(f"    {k:28s} {summary[k]:.3f}")
    p = f"{GATES}/gate1_raw/offset_probe.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
