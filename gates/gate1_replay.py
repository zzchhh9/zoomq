#!/usr/bin/env python3
"""Gate 1 -- how much of a demonstrated BiGym trajectory survives being replayed
as a piecewise-linear ``j``-knot skeleton?

ZoomQ commits an action chunk of ``T = 16`` outer control steps in dyadic
temporal order and executes the *skeleton*: the actions at the committed knots
are kept and every other timestep is filled by linear interpolation between its
two neighbouring knots.  The cumulative knot sets are

    j =  2 : t = 0, 15
    j =  3 : t = 0, 7, 15
    j =  5 : t = 0, 3, 7, 11, 15
    j =  9 : t = 0, 1, 3, 5, 7, 9, 11, 13, 15
    j = 16 : every t                       (identity -- the control)

If a coarse skeleton still completes ``move_plate``, an anytime policy that
stops descending early is viable; if everything collapses at ``j < 16`` the
anytime idea is falsified.

MODES
-----
``raw``     Step 1.  Replay the unmodified demo actions.  ``--era-sweep`` runs
            the era-localisation grid (success_hold_seconds / init_pelvis_z /
            base_action_mode) that tells you whether a 0% here is the demos,
            the physics version or the env era pin.
``full``    Step 2.  Restore the demo's initial state (``env.reset(seed=
            demo.seed)``), then execute the WHOLE demo reprojected chunk by
            chunk (consecutive, non-overlapping 16-step chunks) onto ``j``
            knots, open loop.  Success + mean episode reward per ``j``.
``chunk``   Step 3.  Per-chunk tracking error.  The reference rollout (exact
            demo actions) is snapshotted at every chunk boundary; for each
            ``j`` the sim state is restored bit-exactly at that boundary and
            the 16-step skeleton is executed open loop.  Deviation is measured
            against the reference rollout's OWN achieved trajectory, so
            ``j = 16`` must come out at exactly 0.0 -- that is the bit-exact
            restore check, reported as ``restore_bitexact``.
``zoh``     Step 4a.  Same as ``full`` but zero-order hold between knots
            instead of linear interpolation (isolates interpolation from knot
            count).
``perturb`` Step 4b.  Same as ``full`` at a given ``j`` but one knot per chunk
            is displaced by one quantisation quantum, to measure sensitivity.

ERAS
----
``--era homie``     the production CQN-AS/ZoomQ config: HOMIE ``mjlab_h1``
                    lower body, ``base_action_mode=homie_cmd``,
                    ``init_pelvis_z=0.92``, ``reset_warmup_steps=160``,
                    ``success_hold_seconds=1.0``.  Identical to
                    ``cfgs/bigym_task/move_plate.yaml`` + ``default.yaml``.
``--era floating``  the era the stock 0.9.0 demos were RECORDED in: no
                    lower-body policy, the pelvis is a kinematic floating base
                    driven by the demo's own ``legacy_delta`` base slots,
                    ``success_hold_seconds=0``.

Everything is read-only: demos are loaded through the BiGym ``DemoStore`` cache
and the repo's own ``convert_demo_to_timesteps`` / ``extract_action_stats`` /
``_legacy_delta_action_to_homie_cmd`` conversion path, so the actions executed
here are bit-identical to the ones a CQN-AS/ZoomQ run trains on.

Deterministic: each demo is replayed from its own recorded reset seed
(``demo.seed``), demos are ordered by that seed (``DemoStore`` itself returns
them shuffled), and nothing else draws from an RNG.
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
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

CHUNK_LEN = 16
J_VALUES = (2, 3, 5, 9, 16)

# Per-task env facts, copied from cfgs/bigym_task/<task>.yaml of the repo
# (the trainer reads exactly these).  ``enable_all_floating_dof`` changes the
# floating-DOF set and therefore the demo directory ('..._pelvis_z_...'), so
# it MUST match the yaml or DemoStore returns a different demo family.
TASK_CONFIG = {
    "move_plate":       {"enable_all_floating_dof": False,
                         "episode_length": 3000, "n_demos": 60},
    "drawer_top_close": {"enable_all_floating_dof": True,
                         "episode_length": 3000, "n_demos": 51},
    # Phase 2 primary band. episode_length and enable_all_floating_dof copied
    # from cfgs/bigym_task/<task>.yaml -- they select the demo directory, so a
    # mismatch silently returns a different demo family.
    "take_cups":        {"enable_all_floating_dof": True,
                         "episode_length": 10500, "n_demos": 36},
    "put_cups":         {"enable_all_floating_dof": True,
                         "episode_length": 8500, "n_demos": 56},
}

# CQN-AS quantisation actually used by the live runs (.hydra/config.yaml:
# levels: 3, bins: 5 over the normalised [-1, 1] action range).
FINE_BIN_QUANTUM = 2.0 / (5 ** 3)          # 0.016
# ZoomQ residual-window half-width for a round-1 knot (w_schedule in
# bigym_src/zoomq.py: 1.0, 0.15, 0.08, 0.04, 0.02).
WINDOW_QUANTUM = 0.15


# --------------------------------------------------------------------------- #
# dyadic knots
# --------------------------------------------------------------------------- #
def dyadic_rounds(chunk_len: int) -> list[list[int]]:
    committed = [0, chunk_len - 1]
    rounds = [sorted(committed)]
    while len(committed) < chunk_len:
        ordered = sorted(committed)
        new = [(lo + hi) // 2 for lo, hi in zip(ordered[:-1], ordered[1:])
               if hi - lo >= 2]
        if not new:
            break
        rounds.append(sorted(new))
        committed.extend(new)
    return rounds


def knots_for_j(j: int, chunk_len: int = CHUNK_LEN) -> list[int]:
    """Cumulative dyadic knot set of size j."""
    out: list[int] = []
    for rnd in dyadic_rounds(chunk_len):
        out.extend(rnd)
        if len(out) >= j:
            break
    if len(out) != j:
        raise ValueError(f"j={j} is not a dyadic knot count for T={chunk_len} "
                         f"(valid: {[len(k) for k in _cumulative(chunk_len)]})")
    return sorted(out)


def _cumulative(chunk_len: int) -> list[list[int]]:
    out, acc = [], []
    for rnd in dyadic_rounds(chunk_len):
        acc = sorted(acc + rnd)
        out.append(list(acc))
    return out


def project(chunk: np.ndarray, knots: list[int], kind: str = "linear",
            hold_dims: tuple = ()) -> np.ndarray:
    """Keep the actions at ``knots``; fill the rest.

    ``linear``: linear interpolation between neighbouring knots.
    ``zoh``   : zero-order hold (the previous knot's value).

    ``hold_dims`` overrides ``kind`` for those dims only, giving them the
    zero-order hold.  This mirrors ``bigym_src/zoomq.py::_fill``, which returns
    ``lo`` (the left knot) for every dim listed in ``zoomq.hold_dims`` and the
    linear blend for the rest -- so a replay under ``hold_dims`` executes
    exactly the chunks a ZoomQ arm configured that way would execute.
    """
    m = chunk.shape[0]
    ks = sorted({0, m - 1} | {k for k in knots if k < m})
    out = np.empty_like(chunk)
    grid = np.arange(m, dtype=np.float64)
    ka = np.asarray(ks)
    zidx = np.searchsorted(ks, grid, side="right") - 1
    for d in range(chunk.shape[1]):
        if d in hold_dims:
            out[:, d] = chunk[ka[zidx], d]
        elif kind == "linear":
            out[:, d] = np.interp(grid, ks, chunk[ks, d])
        elif kind == "zoh":
            out[:, d] = chunk[ka[zidx], d]
        else:
            raise ValueError(kind)
    return out


def project_sequence(actions: np.ndarray, j: int, kind: str = "linear",
                     chunk_len: int = CHUNK_LEN, hold_dims: tuple = (),
                     phase: int = 0) -> np.ndarray:
    """Project a whole action sequence, chunk by chunk (non-overlapping).

    ``phase`` slides the CHUNK GRID without changing which actions are
    executed or how many: boundaries land on ``phase, phase+K, phase+2K, ...``
    and the leading ``phase`` actions form one short chunk.  A deployed agent
    does not choose its phase relative to the task -- the grid is set by when
    the episode happens to start and when it replans -- so sweeping this is the
    only way to read how much of a skeleton's success is the skeleton and how
    much is a lucky alignment between a knot and the moment that matters.
    """
    knots = knots_for_j(j, chunk_len)
    out = np.empty_like(actions)
    ph = int(phase) % chunk_len
    starts = ([0] if ph else []) + list(range(ph, len(actions), chunk_len))
    for s in starts:
        e = min(s + (chunk_len if s or not ph else ph), len(actions))
        if e <= s:
            continue
        out[s:e] = project(actions[s:e], knots, kind, hold_dims)
    return out


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
HOMIE_POLICY = {
    "enabled": True,
    "backend": "mjlab_h1",
    "checkpoint_path": f"{REPO}/auto-rs-best-fix.pt",
    "device": "cpu",
    "support_base_with_legs": True,
    "base_action_mode": "homie_cmd",
    "route_pelvis_rz_to_torso": True,
    "reset_warmup_steps": 160,
    "default_height_cmd": 0.98,
    "init_pelvis_z": 0.92,
    "use_height_cmd": True,
    "cmd_clip": 1.0,
    "wz_clip": 1.0,
    "action_clip": 5.0,
}
FLOATING_POLICY = {
    "enabled": False,
    "base_action_mode": "legacy_delta",
    "support_base_with_legs": False,
}

ERAS = {
    # name: (policy dict, default success_hold_seconds)
    "homie": (HOMIE_POLICY, 1.0),
    "floating": (FLOATING_POLICY, 0.0),
}

# Step-1 era-localisation grid (see the module docstring).
ERA_VARIANTS = {
    "homie_prod_hold1.0_z0.92":      ("homie", {}, 1.0),
    "homie_hold0_z0.92":             ("homie", {}, 0.0),
    "homie_hold1.0_z1.00":           ("homie", {"init_pelvis_z": 1.00}, 1.0),
    "homie_hold0_z1.00":             ("homie", {"init_pelvis_z": 1.00}, 0.0),
    "homie_hold0_z0.92_warmup0":     ("homie", {"reset_warmup_steps": 0}, 0.0),
    "homie_hold0_z0.92_nortz":       ("homie", {"route_pelvis_rz_to_torso": False}, 0.0),
    "homie_hold0_z0.92_cmdclip5":    ("homie", {"cmd_clip": 5.0, "wz_clip": 5.0}, 0.0),
    "floating_hold1.0":              ("floating", {}, 1.0),
    "floating_hold0":                ("floating", {}, 0.0),
}


def build_env(era: str, hold: float, overrides: dict | None = None,
              task: str = "move_plate", dsr: int = 10):
    """Build the replay env.

    ``dsr`` (``demo_down_sample_rate``) is the ONLY rate knob: bigym derives
    the env control frequency from it as ``CONTROL_FREQUENCY_MAX // dsr``
    (bigym/loco/env.py) and DemoStore decimates the 500 Hz source demos to
    the same frequency, so demo rate and control rate cannot diverge.
    """
    import bigym_src.bigym_env as bigym_env

    tc = TASK_CONFIG[task]
    policy, _ = ERAS[era]
    pol = dict(policy)
    pol.update(overrides or {})
    return bigym_env.make(
        task_name=task,
        enable_all_floating_dof=tc["enable_all_floating_dof"],
        control_pelvis=True,
        action_mode="absolute",
        demo_down_sample_rate=int(dsr),
        episode_length=tc["episode_length"],
        frame_stack=1,
        camera_shape=(84, 84),
        camera_keys=(),           # State demos; no pixels needed for replay
        state_keys=("proprioception", "proprioception_grippers",
                    "proprioception_floating_base"),
        render_mode="rgb_array",
        normalize_low_dim_obs=False,
        lowerbody_policy=pol,
        robot_model="h1",
        success_hold_seconds=hold,
        path_root=REPO,
    )


# --------------------------------------------------------------------------- #
# demos
# --------------------------------------------------------------------------- #
def _trim(steps):
    """The training truncation: drop everything after the first rewarding step."""
    out = []
    for ds in steps:
        out.append(ds)
        if ds.reward > 0:
            break
    return out


def load_demos(env, untrimmed: bool = True, cache_root: str | None = None,
               order: str = "seed", keep_first: bool = False):
    """Load the stock DemoStore demos through the repo's own conversion path.

    Returns a list of dicts with the demo's reset seed and its actions already
    in the normalised [-1, 1] space the critic/replay-buffer sees.  Action
    stats are fitted on the TRIMMED demos, exactly as ``BiGym.get_demos`` does,
    so the normalisation matches a training run bit for bit.
    """
    from demonstrations.demo_store import DemoStore
    from bigym.bigym_env import CONTROL_FREQUENCY_MAX

    inner = env._env
    dsr = int(inner._demo_down_sample_rate)
    # bigym/loco/env.py BiGym.get_demos uses exactly this frequency.
    freq = CONTROL_FREQUENCY_MAX // dsr
    store = DemoStore(Path(cache_root)) if cache_root else DemoStore()
    raw = store.get_demos(inner._demo_metadata(), amount=-1, frequency=freq)
    for d in raw:
        for ts in d.timesteps:
            ts.observation = {k: np.array(v, dtype=np.float32)
                              for k, v in ts.observation.items()}
    # DemoStore._get_demos() ends in np.random.shuffle(files): the order it
    # returns depends on the GLOBAL numpy RNG state, so a positional index
    # into that list is NOT a stable demo key.  The stable keys are the demo
    # uuid (== its .safetensors filename) and its recorded reset seed; both
    # are unique per task.  Rank in either total order is deterministic.
    uuid_rank = {u: i for i, u in
                 enumerate(sorted(d.metadata.uuid for d in raw))}
    seed_rank = {s: i for i, s in enumerate(sorted(int(d.seed) for d in raw))}
    if order == "uuid":
        raw.sort(key=lambda d: d.metadata.uuid)
    else:
        raw.sort(key=lambda d: int(d.seed))

    # training-identical action stats
    converted = [inner.convert_demo_to_timesteps(_trim(d.timesteps))[0] for d in raw]
    inner._action_stats = inner.extract_action_stats(converted)

    out = []
    for d in raw:
        steps = d.timesteps if untrimmed else _trim(d.timesteps)
        acts = []
        for ds in steps:
            a = ds.info["demo_action"].astype(np.float32)
            if inner._uses_homie_cmd_base_actions():
                a = inner._legacy_delta_action_to_homie_cmd(
                    a, floating_qpos=ds.observation.get(
                        "proprioception_floating_base", None))
            acts.append(a)
        acts = inner._convert_action_from_raw(np.stack(acts))
        clipped = float(np.mean((acts < -1) | (acts > 1)))
        # DemoStep.action is "the action taken to REACH the timestep".  The
        # historical Gate-1 default treats index 0 as the reset step's dummy
        # action and executes [1:].  That is WRONG for the cached DemoStore
        # demos: DemoConverter.create_demo_in_new_env resets and then steps
        # with EVERY source action, so its timestep 0 is already a
        # post-action state and carries a real executed action.  Dropping it
        # replays a trajectory one action short (measured cost on move_plate
        # at 50 Hz: 0.867 -> 0.750).  keep_first=True executes all of them.
        out.append({
            "seed": int(d.seed),
            "uuid": str(d.metadata.uuid),
            "index": int(uuid_rank[d.metadata.uuid]),
            "seed_rank": int(seed_rank[int(d.seed)]),
            "actions": np.clip(acts if keep_first else acts[1:],
                               -1, 1).astype(np.float32),
            "clipped_frac": clipped,
            "demo_reward": float(sum(s.reward for s in steps)),
            "demo_steps": int(len(steps) if keep_first else len(steps) - 1),
            "demo_first_reward_step": next(
                (i for i, s in enumerate(steps) if s.reward > 0), -1),
        })
    return out


# --------------------------------------------------------------------------- #
# rollout helpers
# --------------------------------------------------------------------------- #
def _hand_pos(inner) -> np.ndarray:
    from bigym.const import HandSide

    robot = inner._env.robot
    return np.concatenate([np.asarray(robot.get_hand_pos(s), dtype=np.float64)
                           for s in (HandSide.LEFT, HandSide.RIGHT)])


def _proprio(inner) -> np.ndarray:
    obs = inner._env.get_observation()
    p = np.asarray(obs["proprioception"], dtype=np.float64)
    return np.asarray(inner._maybe_crop_proprioception(p), dtype=np.float64)


def rollout(env, actions, seed, record=False):
    """Open-loop rollout of a normalised action sequence from ``seed``."""
    inner = env._env
    env.reset(seed=int(seed))
    rew, steps, hands, joints = 0.0, 0, [], []
    first_rew = -1
    for a in actions:
        ts = env.step(np.asarray(a, dtype=np.float32))
        if first_rew < 0 and float(ts.reward) > 0:
            first_rew = steps
        rew += float(ts.reward)
        steps += 1
        if record:
            hands.append(_hand_pos(inner))
            joints.append(_proprio(inner))
        if ts.last():
            break
    res = {"seed": int(seed), "steps": steps, "reward": rew,
           "first_reward_step": int(first_rew),
           "success": int(rew >= 0.25)}
    if record:
        res["hands"] = np.stack(hands)
        res["joints"] = np.stack(joints)
    return res


def snapshot(inner) -> dict:
    import mujoco  # noqa: F401  (imported for symmetry with restore)

    data = inner._env._mojo.data
    snap = {
        "qpos": np.array(data.qpos, dtype=np.float64),
        "qvel": np.array(data.qvel, dtype=np.float64),
        "ctrl": np.array(data.ctrl, dtype=np.float64),
        "warm": np.array(data.qacc_warmstart, dtype=np.float64),
        "step_counter": int(inner._step_counter),
    }
    try:
        snap["lb"] = inner.get_lowerbody_state()
    except Exception:
        snap["lb"] = None
    return snap


def restore(inner, snap: dict) -> None:
    """Bit-exact mid-episode restore (tests/smoke/snapshot_roundtrip_probe.py)."""
    import mujoco

    model = inner._env._mojo.model
    data = inner._env._mojo.data
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.ctrl[:] = snap["ctrl"]
    mujoco.mj_forward(model, data)
    # mj_forward overwrites qacc_warmstart with the recomputed qacc, which is
    # not bit-identical to the stepping-time warmstart -> put it back last.
    data.qacc_warmstart[:] = snap["warm"]
    if snap.get("lb"):
        inner.set_lowerbody_state(snap["lb"])
    inner._step_counter = int(snap["step_counter"])


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_raw(env, demos, args):
    """Replay the unmodified demo actions.  Rows carry the demo identity so
    the verdict is attributable to a specific demonstration file."""
    rows = []
    for d in demos:
        r = rollout(env, d["actions"], d["seed"])
        for k in ("uuid", "index", "seed_rank", "clipped_frac",
                  "demo_reward", "demo_steps", "demo_first_reward_step"):
            if k in d:
                r[k] = d[k]
        rows.append(r)
    return {"rows": rows, **_agg(rows)}


def mode_full(env, demos, args, kind="linear"):
    out = {}
    for j in args.j:
        rows = []
        for d in demos:
            src = d["actions"]
            sigma = float(getattr(args, "knot_noise", 0.0) or 0.0)
            if sigma > 0.0:
                # THE ERROR BUDGET. Gate 1's spec asks for this and it was never
                # run: "re-execute with knots perturbed ... since the trained
                # critic will not commit DP-optimal skeletons; the resulting
                # cliff is the error budget the critic must live inside."
                #
                # Perturb EVERY knot (not one per chunk, as --mode perturb does)
                # by N(0, sigma) before projecting, so the interpolant is built
                # from the perturbed knots exactly as a trained critic's would
                # be. Reference points: a snapshot probe measured selected-knot
                # MAE against the demonstrations at 0.049 for ZoomQ and 0.008 for
                # the CQN-AS baseline, so sweeping sigma across that range says
                # whether ZoomQ's knot precision alone can explain its zero.
                # Seeded from the demo's own reset seed so a sweep is repeatable.
                rng = np.random.default_rng(args.seed * 7919 + d["seed"])
                src = src.copy()
                kn = knots_for_j(j, args.chunk_len)
                for st in range(0, len(src), args.chunk_len):
                    for k in kn:
                        t = st + k
                        if t < len(src):
                            src[t] = src[t] + rng.normal(0.0, sigma, size=src.shape[1])
                src = np.clip(src, -1.0, 1.0)
            acts = project_sequence(src, j, kind, args.chunk_len,
                                    tuple(getattr(args, "hold_dims", ()) or ()),
                                    phase=int(getattr(args, "phase", 0) or 0))
            # THE EXACT PREFIX. The phase sweep showed success is set by whether
            # the first few control steps after reset are reproduced exactly
            # (phases 1-5, which give a short dense leading chunk, score
            # 0.78-0.88 at j=3; phases 0 and 6-14, which do not, score
            # 0.32-0.57) -- and not by any periodic alignment. This isolates
            # that directly: execute the first N demo actions verbatim and
            # project everything after them.
            npre = int(getattr(args, "exact_prefix", 0) or 0)
            if npre > 0:
                acts = acts.copy()
                acts[:npre] = src[:npre]
            r = rollout(env, np.clip(acts, -1, 1), d["seed"])
            r["skeleton_rmse"] = float(np.sqrt(np.mean(
                (acts - d["actions"]) ** 2)))
            rows.append(r)
        out[str(j)] = {"rows": rows,
                       "knot_noise": float(getattr(args, "knot_noise", 0.0) or 0.0),
                       "phase": int(getattr(args, "phase", 0) or 0),
                       "exact_prefix": int(getattr(args, "exact_prefix", 0) or 0),
                       **_agg(rows)}
        print(f"    j={j:2d} success={out[str(j)]['success_rate']:.3f} "
              f"mean_reward={out[str(j)]['mean_reward']:.3f} "
              f"skeleton_rmse={out[str(j)]['mean_skeleton_rmse']:.4f}", flush=True)
    return out


def mode_perturb(env, demos, args):
    """Displace one knot per chunk by one quantisation quantum, then run ``full``.

    The perturbed knot is the middle one of the chunk's knot set (the knot
    committed in the deepest round of that ``j``).  Three sign models:

      plus / minus  every action dimension of that knot is shifted by the same
                    signed quantum -- a worst-case DC bias;
      rand          each (chunk, dimension) is rounded up or down
                    independently, which is what a real quantiser does.
                    Seeded from the demo's own reset seed, so it repeats.

    ``fine_bin`` = 2/5^3 = 0.016 is the CQN-AS decoder's finest bin over the
    normalised [-1, 1] range (levels=3, bins=5 in the live ZoomQ runs);
    ``window`` = 0.15 is ZoomQ's residual-window half-width for a round-1 knot.
    """
    out = {}
    for j in args.j:
        knots = knots_for_j(j, args.chunk_len)
        # the knot committed last (deepest round) is the sensitive one
        target = knots[len(knots) // 2]
        for qname, q in (("fine_bin", FINE_BIN_QUANTUM),
                         ("window", WINDOW_QUANTUM)):
            for sign in ("plus", "minus", "rand"):
                rows = []
                for d in demos:
                    a = d["actions"].copy()
                    rng = np.random.default_rng(args.seed * 100003 + d["seed"])
                    for s in range(0, len(a), args.chunk_len):
                        t = s + target
                        if t >= len(a):
                            continue
                        if sign == "plus":
                            delta = q
                        elif sign == "minus":
                            delta = -q
                        else:
                            delta = q * rng.choice([-1.0, 1.0], size=a.shape[1])
                        a[t] = a[t] + delta
                    acts = np.clip(project_sequence(
                        np.clip(a, -1, 1), j, "linear", args.chunk_len), -1, 1)
                    rows.append(rollout(env, acts, d["seed"]))
                key = f"j{j}_{qname}_{sign}"
                out[key] = {"knot": int(target), "quantum": float(q),
                            "rows": rows, **_agg(rows)}
                print(f"    {key}: success={out[key]['success_rate']:.3f}",
                      flush=True)
    return out


def mode_chunk(env, demos, args):
    """Per-chunk open-loop tracking error against the reference rollout."""
    inner = env._env
    per_j = {str(j): [] for j in args.j}
    for d in demos:
        acts = d["actions"]
        n_chunks = len(acts) // args.chunk_len       # whole chunks only
        if n_chunks == 0:
            continue
        # --- reference rollout, snapshotting each chunk boundary -----------
        env.reset(seed=int(d["seed"]))
        snaps, ref_hands, ref_joints, alive = [], [], [], True
        for c in range(n_chunks):
            snaps.append(snapshot(inner))
            h, jt = [], []
            for t in range(args.chunk_len):
                ts = env.step(acts[c * args.chunk_len + t])
                h.append(_hand_pos(inner))
                jt.append(_proprio(inner))
                if ts.last():
                    alive = False
                    break
            ref_hands.append(np.stack(h))
            ref_joints.append(np.stack(jt))
            if not alive:
                break
        n_chunks = len(ref_hands)
        # --- per-j restore + skeleton --------------------------------------
        for j in args.j:
            knots = knots_for_j(j, args.chunk_len)
            for c in range(n_chunks):
                chunk = acts[c * args.chunk_len:(c + 1) * args.chunk_len]
                sk = np.clip(project(chunk, knots, "linear"), -1, 1)
                restore(inner, snaps[c])
                h, jt = [], []
                for t in range(len(ref_hands[c])):
                    env.step(sk[t].astype(np.float32))
                    h.append(_hand_pos(inner))
                    jt.append(_proprio(inner))
                h, jt = np.stack(h), np.stack(jt)
                rh, rj = ref_hands[c], ref_joints[c]
                ee = np.stack([
                    np.linalg.norm(h[:, :3] - rh[:, :3], axis=1),
                    np.linalg.norm(h[:, 3:] - rh[:, 3:], axis=1)]).mean(0)
                per_j[str(j)].append({
                    "seed": int(d["seed"]), "chunk": c,
                    "ee_final_m": float(ee[-1]),
                    "ee_max_m": float(ee.max()),
                    "joint_final_rad": float(np.abs(jt[-1] - rj[-1]).mean()),
                    "joint_max_rad": float(np.abs(jt - rj).max()),
                    "skeleton_rmse": float(np.sqrt(np.mean((sk - chunk) ** 2))),
                })
    out = {}
    for j in args.j:
        rows = per_j[str(j)]
        agg = {"n_chunks": len(rows)}
        for k in ("ee_final_m", "ee_max_m", "joint_final_rad", "joint_max_rad",
                  "skeleton_rmse"):
            v = np.array([r[k] for r in rows], dtype=np.float64)
            agg[f"mean_{k}"] = float(v.mean()) if len(v) else float("nan")
            agg[f"p90_{k}"] = float(np.percentile(v, 90)) if len(v) else float("nan")
        out[str(j)] = {"rows": rows, **agg}
        print(f"    j={j:2d} n={agg['n_chunks']} "
              f"ee_final mean={agg['mean_ee_final_m']:.4f} m "
              f"p90={agg['p90_ee_final_m']:.4f} m  "
              f"ee_max mean={agg['mean_ee_max_m']:.4f} m", flush=True)
    j16 = out.get("16")
    if j16 is not None:
        out["restore_bitexact"] = {
            "max_ee_dev_m": float(max((r["ee_max_m"] for r in j16["rows"]),
                                      default=float("nan"))),
            "max_joint_dev_rad": float(max((r["joint_max_rad"] for r in j16["rows"]),
                                           default=float("nan"))),
            "note": "j=16 replays the reference actions after a restore; a "
                    "non-zero value here means the restore is NOT bit-exact "
                    "and every other number in this mode is suspect.",
        }
    return out


def _agg(rows):
    if not rows:
        return {"n": 0, "success_rate": float("nan"), "mean_reward": float("nan")}
    out = {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
    }
    if "skeleton_rmse" in rows[0]:
        out["mean_skeleton_rmse"] = float(np.mean(
            [r["skeleton_rmse"] for r in rows]))
    return out


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def _worker(payload):
    args, shard, era, overrides, hold = payload
    np.random.seed(args.seed)
    env = build_env(era, hold, overrides,
                    task=getattr(args, "task", "move_plate"),
                    dsr=int(getattr(args, "dsr", 10)))
    demos = load_demos(env, untrimmed=args.untrimmed,
                       cache_root=getattr(args, "cache_root", None),
                       order=getattr(args, "order", "seed"),
                       keep_first=bool(getattr(args, "keep_first", False)))
    demos = [demos[i] for i in shard]
    try:
        if args.mode == "raw":
            res = mode_raw(env, demos, args)
        elif args.mode == "full":
            res = mode_full(env, demos, args, "linear")
        elif args.mode == "zoh":
            res = mode_full(env, demos, args, "zoh")
        elif args.mode == "chunk":
            res = mode_chunk(env, demos, args)
        elif args.mode == "perturb":
            res = mode_perturb(env, demos, args)
        else:
            raise ValueError(args.mode)
    finally:
        env.close()
    return res


def _merge(parts):
    """Concatenate the ``rows`` lists of shard results and re-aggregate."""
    if not parts:
        return {}
    if "rows" in parts[0]:
        rows = [r for p in parts for r in p["rows"]]
        return {"rows": rows, **_agg(rows)} if "success" in (rows[0] if rows else {}) \
            else _merge_chunk_rows(rows)
    out = {}
    for k in parts[0]:
        out[k] = _merge([p[k] for p in parts if k in p])
    return out


def _merge_chunk_rows(rows):
    agg = {"n_chunks": len(rows)}
    for k in ("ee_final_m", "ee_max_m", "joint_final_rad", "joint_max_rad",
              "skeleton_rmse"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        agg[f"mean_{k}"] = float(v.mean()) if len(v) else float("nan")
        agg[f"p90_{k}"] = float(np.percentile(v, 90)) if len(v) else float("nan")
    return {"rows": rows, **agg}


def run_config(args, era, overrides, hold, n_demos):
    shards = [list(range(i, n_demos, args.workers)) for i in range(args.workers)]
    shards = [s for s in shards if s]
    payloads = [(args, s, era, overrides, hold) for s in shards]
    if len(payloads) == 1:
        parts = [_worker(payloads[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            parts = pool.map(_worker, payloads)
    if args.mode == "chunk":
        merged = {}
        for j in args.j:
            merged[str(j)] = _merge_chunk_rows(
                [r for p in parts for r in p[str(j)]["rows"]])
        j16 = merged.get("16")
        if j16 is not None:
            merged["restore_bitexact"] = {
                "max_ee_dev_m": float(max((r["ee_max_m"] for r in j16["rows"]),
                                          default=float("nan"))),
                "max_joint_dev_rad": float(max(
                    (r["joint_max_rad"] for r in j16["rows"]),
                    default=float("nan"))),
            }
        return merged
    return _merge(parts)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="full",
                   choices=("raw", "full", "chunk", "zoh", "perturb"),
                   help="which Gate-1 step to run (see the module docstring)")
    p.add_argument("--era", default="floating", choices=tuple(ERAS),
                   help="env era: 'homie' = production ZoomQ config, "
                        "'floating' = the era the stock demos were recorded in")
    p.add_argument("--hold", type=float, default=None,
                   help="success_hold_seconds override (default: era default)")
    p.add_argument("--task", default="move_plate", choices=tuple(TASK_CONFIG),
                   help="BiGym task; picks up enable_all_floating_dof and "
                        "episode_length from cfgs/bigym_task/<task>.yaml")
    p.add_argument("--dsr", type=int, default=10,
                   help="demo_down_sample_rate. Sets BOTH the demo decimation "
                        "frequency and the env control frequency "
                        "(= 500 // dsr Hz); they cannot diverge. 500//dsr must "
                        "stay in [20, 500], i.e. dsr in [1, 25].")
    p.add_argument("--cache-root", default=None,
                   help="private DemoStore cache root. Any frequency that is "
                        "not cached yet is REGENERATED and written under this "
                        "root; leave unset to use ~/.bigym (read-only in this "
                        "project -- pass a private root before sweeping dsr).")
    p.add_argument("--order", default="seed", choices=("seed", "uuid"),
                   help="deterministic demo order used for sharding. Neither "
                        "is DemoStore's own order (it shuffles); the reported "
                        "per-row 'index' is always the uuid rank.")
    p.add_argument("--keep-first", action="store_true", default=False,
                   help="execute the demo's action[0] too. The default "
                        "(off) reproduces the published Gate-1 numbers, "
                        "which drop it; the cached demos have no dummy reset "
                        "action, so dropping it replays one action short.")
    p.add_argument("--num-demos", type=int, default=20,
                   help="number of demos to replay (max 60 for move_plate)")
    p.add_argument("--j", type=str, default="2,3,5,9,16",
                   help="comma-separated dyadic knot counts")
    p.add_argument("--chunk-len", type=int, default=CHUNK_LEN)
    p.add_argument("--exact-prefix", type=int, default=0,
                   help="execute the first N demo actions verbatim, project the rest")
    p.add_argument("--phase", type=int, default=0,
                   help="slide the chunk grid by this many steps (0..K-1); "
                        "changes alignment only, not which actions execute")
    p.add_argument("--knot-noise", type=float, default=0.0,
                   help="stddev of Gaussian noise added to EVERY knot before "
                        "projection, in normalised action units. The error "
                        "budget sweep Gate 1 specifies. Reference: measured "
                        "selected-knot MAE is 0.049 for ZoomQ and 0.008 for the "
                        "CQN-AS baseline.")
    p.add_argument("--hold-dims", type=str, default="",
                   help="comma-separated action dims that get a ZERO-ORDER "
                        "HOLD instead of the linear fill, mirroring "
                        "zoomq.hold_dims. The binary gripper dims are 13 for "
                        "move_plate (D=15) and 14,15 for the D=16 tasks; a "
                        "linear fill there commands a half-closed gripper the "
                        "demonstrations never contain.")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel worker processes (each builds its own env)")
    p.add_argument("--untrimmed", action="store_true", default=True,
                   help="replay the full recorded demo (default). The training "
                        "loader truncates at the first rewarding step, which "
                        "leaves zero slack for a success-hold.")
    p.add_argument("--trimmed", dest="untrimmed", action="store_false",
                   help="truncate at the first rewarding step, as training does")
    p.add_argument("--era-sweep", action="store_true",
                   help="mode=raw only: run the whole era-localisation grid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write JSON here")
    args = p.parse_args()
    args.j = [int(x) for x in str(args.j).split(",") if x]
    args.hold_dims = tuple(int(x) for x in str(args.hold_dims).split(",") if x.strip())

    t0 = time.time()
    n = int(args.num_demos)
    result = {"config": {k: v for k, v in vars(args).items()},
              "fine_bin_quantum": FINE_BIN_QUANTUM,
              "window_quantum": WINDOW_QUANTUM,
              "knot_sets": {str(j): knots_for_j(j, args.chunk_len)
                            for j in args.j}}

    if args.mode == "raw" and args.era_sweep:
        for name, (era, over, hold) in ERA_VARIANTS.items():
            print(f"[{name}]", flush=True)
            r = run_config(args, era, over, hold, n)
            result[name] = r
            print(f"  success={r['success_rate']:.3f} "
                  f"mean_reward={r['mean_reward']:.3f} n={r['n']}", flush=True)
    else:
        hold = args.hold if args.hold is not None else ERAS[args.era][1]
        print(f"[{args.mode} era={args.era} hold={hold} "
              f"untrimmed={args.untrimmed} n={n}]", flush=True)
        result[f"{args.mode}_{args.era}"] = run_config(args, args.era, {}, hold, n)

    result["wall_seconds"] = time.time() - t0
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print("wrote", args.out, f"({result['wall_seconds']:.0f}s)")
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "config"},
                         indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
