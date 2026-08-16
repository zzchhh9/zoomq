#!/usr/bin/env python3
"""Gate-1 follow-up probe: demo identity + compiled physics rate.

Answers, from the RUNTIME objects (not yaml):
  * the compiled MuJoCo ``opt.timestep`` of the scene each task actually loads;
  * the substep count one outer control step takes, and the resulting
    ``control_step_seconds``;
  * the ``control_frequency`` the loco wrapper derives from
    ``demo_down_sample_rate``;
  * every metadata field the stock ``.safetensors`` demos carry (does anything
    record the rate they were RECORDED at?);
  * whether ``DemoStore`` returns demos in a stable order (it does not --
    ``_get_demos`` calls ``np.random.shuffle``), and what a stable key is.

Read-only.  Writes nothing.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.environ.get("GATE1_REPO", "/mnt/workspace/zoomq/third_party/CQN-AS-G1")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

TASKS = {
    # task_name: enable_all_floating_dof (from cfgs/bigym_task/<task>.yaml)
    "move_plate": False,
    "drawer_top_close": True,
}


def build(task, all_dof, dsr):
    import bigym_src.bigym_env as bigym_env

    return bigym_env.make(
        task_name=task,
        enable_all_floating_dof=all_dof,
        control_pelvis=True,
        action_mode="absolute",
        demo_down_sample_rate=dsr,
        episode_length=3000,
        frame_stack=1,
        camera_shape=(84, 84),
        camera_keys=(),
        state_keys=("proprioception", "proprioception_grippers",
                    "proprioception_floating_base"),
        render_mode="rgb_array",
        normalize_low_dim_obs=False,
        lowerbody_policy={"enabled": False,
                          "base_action_mode": "legacy_delta",
                          "support_base_with_legs": False},
        robot_model="h1",
        success_hold_seconds=0.0,
        path_root=REPO,
    )


def main():
    out = {}
    import bigym.bigym_env as bg

    out["constants"] = {
        "CONTROL_FREQUENCY_MAX": int(bg.CONTROL_FREQUENCY_MAX),
        "CONTROL_FREQUENCY_MIN": int(bg.CONTROL_FREQUENCY_MIN),
        "PHYSICS_DT_nominal": float(bg.PHYSICS_DT),
    }
    import mujoco
    out["versions"] = {"mujoco": mujoco.__version__}

    for task, all_dof in TASKS.items():
        rec = {}
        for dsr in (5, 10, 25):
            env = build(task, all_dof, dsr)
            inner = env._env          # loco.BiGym
            raw = inner._env          # bigym.BiGymEnv
            model = raw._mojo.model
            rec[f"dsr{dsr}"] = {
                "compiled_timestep_s": float(model.opt.timestep),
                "nominal_PHYSICS_DT_s": float(bg.PHYSICS_DT),
                "control_frequency_hz": int(raw.control_frequency),
                "sub_steps_count": int(raw._sub_steps_count),
                "control_step_seconds": float(raw.control_step_seconds),
                "effective_hz": 1.0 / float(raw.control_step_seconds),
                "success_hold_steps_at_1s": int(
                    np.round(1.0 / float(raw.control_step_seconds))),
                "model_path": str(raw._MODEL_PATH),
                "max_outer_steps": 3000 // dsr,
                "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
                "solver": int(model.opt.solver),
                "integrator": int(model.opt.integrator),
            }
            if dsr == 10:
                # ---- demo metadata -------------------------------------- #
                from demonstrations.demo_store import DemoStore
                md = inner._demo_metadata()
                store = DemoStore()
                paths = sorted(p.name for p in store.list_demo_paths(md))
                rec["metadata_repr"] = str(md)
                rec["light_demo_files_n"] = len(paths)
                demos = store.get_demos(md, amount=-1, frequency=50)
                rec["n_demos"] = len(demos)
                d0 = demos[0]
                m0 = d0.metadata
                rec["demo_metadata_fields"] = sorted(
                    k for k in vars(m0) if not k.startswith("__"))
                rec["demo0"] = {
                    "uuid": m0.uuid,
                    "seed": int(m0.seed),
                    "date": str(getattr(m0, "date", None)),
                    "package_versions": dict(m0.package_versions),
                    "observation_mode": str(m0.observation_mode),
                    "n_timesteps": len(d0.timesteps),
                    "env_data": {k: str(v)[:200] for k, v in
                                 vars(m0.environment_data).items()},
                }
                rec["demo_step_fields"] = sorted(
                    k for k in vars(d0.timesteps[1]) if not k.startswith("__"))
                rec["demo_step1_info_keys"] = sorted(
                    d0.timesteps[1].info.keys())
                # any field anywhere mentioning rate/freq/dt/hz?
                blob = json.dumps({
                    "meta": {k: str(v) for k, v in vars(m0).items()},
                    "envdata": {k: str(v) for k, v in
                                vars(m0.environment_data).items()},
                }).lower()
                rec["rate_field_hits"] = sorted(
                    {w for w in ("control_step_seconds", "frequency", "hz",
                                 "timestep", "dt", "rate", "decimat")
                     if w in blob})
                # package_versions across ALL demos (are they uniform?)
                pv = {}
                seeds, uuids, lens = [], [], []
                for d in demos:
                    pv[json.dumps(d.metadata.package_versions, sort_keys=True)] = \
                        pv.get(json.dumps(d.metadata.package_versions,
                                          sort_keys=True), 0) + 1
                    seeds.append(int(d.metadata.seed))
                    uuids.append(d.metadata.uuid)
                    lens.append(len(d.timesteps))
                rec["package_versions_histogram"] = pv
                rec["dates"] = sorted({str(d.metadata.date) for d in demos})
                rec["n_unique_seeds"] = len(set(seeds))
                rec["n_unique_uuids"] = len(set(uuids))
                rec["timesteps_len_at_50hz"] = {
                    "min": int(min(lens)), "max": int(max(lens)),
                    "mean": float(np.mean(lens))}

                # ---- ordering stability --------------------------------- #
                np.random.seed(0)
                o1 = [d.metadata.uuid for d in
                      store.get_demos(md, amount=-1, frequency=50)]
                np.random.seed(0)
                o2 = [d.metadata.uuid for d in
                      store.get_demos(md, amount=-1, frequency=50)]
                np.random.seed(1)
                o3 = [d.metadata.uuid for d in
                      store.get_demos(md, amount=-1, frequency=50)]
                rec["order_same_np_seed_0_vs_0"] = (o1 == o2)
                rec["order_np_seed_0_vs_1"] = (o1 == o3)
                rec["order_seed0_head"] = o1[:5]
                rec["order_seed1_head"] = o3[:5]
                rec["order_is_sorted_by_uuid"] = (o1 == sorted(o1))
            env.close()
        out[task] = rec

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
