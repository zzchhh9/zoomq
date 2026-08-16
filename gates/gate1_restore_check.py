#!/usr/bin/env python3
"""Bit-exactness of the Gate-1 harness's mid-episode restore, channel by channel."""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/workspace/zoomq/gates")
import numpy as np  # noqa: E402

import gate1_replay as G  # noqa: E402

N_DEMOS = int(os.environ.get("N", "10"))
ERA = os.environ.get("ERA", "floating")


def main():
    env = G.build_env(ERA, G.ERAS[ERA][1], {})
    inner = env._env
    demos = G.load_demos(env, untrimmed=True)[:N_DEMOS]
    nq = len(inner._env.robot.qpos)
    rows = []
    for d in demos:
        acts = d["actions"]
        nc = len(acts) // 16
        env.reset(seed=d["seed"])
        snaps, ref, alive = [], [], True
        for c in range(nc):
            snaps.append(G.snapshot(inner))
            seg = []
            for t in range(16):
                ts = env.step(acts[c * 16 + t])
                seg.append(np.concatenate([
                    np.array(inner._env._mojo.data.qpos, dtype=np.float64),
                    np.array(inner._env._mojo.data.qvel, dtype=np.float64),
                    G._hand_pos(inner)]))
                if ts.last():
                    alive = False
                    break
            ref.append(np.stack(seg))
            if not alive:
                break
        nq_full = len(inner._env._mojo.data.qpos)
        nv_full = len(inner._env._mojo.data.qvel)
        for c in range(len(ref)):
            G.restore(inner, snaps[c])
            seg = []
            for t in range(len(ref[c])):
                env.step(acts[c * 16 + t])
                seg.append(np.concatenate([
                    np.array(inner._env._mojo.data.qpos, dtype=np.float64),
                    np.array(inner._env._mojo.data.qvel, dtype=np.float64),
                    G._hand_pos(inner)]))
            e = np.abs(np.stack(seg) - ref[c])
            qp = e[:, :nq_full]
            rows.append({
                "seed": d["seed"], "chunk": c,
                "qpos_argmax": int(np.unravel_index(qp.argmax(), qp.shape)[1]),
                "qpos": float(qp.max()),
                "qvel": float(e[:, nq_full:nq_full + nv_full].max()),
                "hand_m": float(e[:, -6:].max()),
            })
    env.close()
    out = {
        "era": ERA, "n_demos": len(demos), "n_chunks": len(rows),
        "max_qpos": max(r["qpos"] for r in rows),
        "max_qvel": max(r["qvel"] for r in rows),
        "max_hand_m": max(r["hand_m"] for r in rows),
        "frac_chunks_exact": float(np.mean(
            [r["qpos"] == 0.0 and r["qvel"] == 0.0 for r in rows])),
        "robot_nq": nq,
        "nq_full": nq_full,
        "qpos_argmax_hist": {str(k): int(v) for k, v in
                             zip(*np.unique([r["qpos_argmax"] for r in rows],
                                            return_counts=True))},
    }
    print(json.dumps(out, indent=2))
    with open(f"restore_check_{ERA}.json", "w") as f:
        json.dump({"summary": out, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
