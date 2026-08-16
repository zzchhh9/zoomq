# Gate 1 — does a `j`-knot skeleton still do the task?

**Task**: `move_plate` (H1), stock BiGym `DemoStore` 0.9.0 demos, all 60 episodes.
**Harness**: `gates/gate1_replay.py`. **Numbers**: `gates/gate1_results.json`.
**Date**: 2026-08-16. Everything below is simulator replay; no policy is involved.

---

## Verdict

# PASS

`j = 3` (knots at t = 0, 7, 15 — three of sixteen timesteps) reaches
**0.817** success against the unprojected `j = 16` control's **0.750**
(n = 60 demos each), i.e. **109 % of the control**, far above the pre-registered
90 % bar. `j = 5` reaches 102 % and `j = 9` reaches 98 %. Only `j = 2` falls
below the bar, at 62 %.

The depth floor is therefore **between 2 and 3 knots**: an anytime ZoomQ policy
that stops after round 1 (3 knots committed) executes `move_plate` as well as
one that commits all 16 timesteps. There is no gradation around contact — the
per-chunk tracking error is *largest* in the mid-episode reach/transport phase
and *smallest* in the late placement phase (section 3b), so this is not a GRADED
result.

**This verdict is measured on the `floating` era, not on the production ZoomQ
env, because the production env cannot replay its own demos at all** (0/60,
section 1). That caveat is load-bearing; read section 1 before quoting the
verdict.

---

## 1. Step 1 — raw demo replay (MANDATORY control)

Replay the unmodified demo actions from the demo's own reset seed
(`env.reset(seed=demo.seed)`), through the repo's own conversion path
(`convert_demo_to_timesteps` / `extract_action_stats` /
`_legacy_delta_action_to_homie_cmd`). Success = `episode_reward >= 0.25`, the
repo's own eval predicate. n = 60 demos (all of them) per row.

| variant | lower body | `success_hold_seconds` | `init_pelvis_z` | success |
|---|---|---|---|---|
| **`homie_prod_hold1.0_z0.92`** (production) | HOMIE `mjlab_h1` | 1.0 | 0.92 | **0.000** |
| `homie_hold0_z0.92` | HOMIE | 0.0 | 0.92 | 0.017 |
| `homie_hold1.0_z1.00` | HOMIE | 1.0 | 1.00 | 0.000 |
| `homie_hold0_z1.00` | HOMIE | 0.0 | 1.00 | 0.000 |
| `homie_hold0_z0.92_warmup0` (`reset_warmup_steps=0`) | HOMIE | 0.0 | 0.92 | 0.067 |
| `homie_hold0_z0.92_nortz` (`route_pelvis_rz_to_torso=false`) | HOMIE | 0.0 | 0.92 | 0.000 |
| `homie_hold0_z0.92_cmdclip5` (`cmd_clip=wz_clip=5`) | HOMIE | 0.0 | 0.92 | 0.100 |
| `floating_hold1.0` | none (kinematic base) | 1.0 | — | 0.000 |
| **`floating_hold0`** (the demos' recording era) | none (kinematic base) | 0.0 | — | **0.750** |

**The production config scores 0.000. This is an era mismatch, not an algorithm
failure, and it has two independent causes — neither one alone rescues it.**

### Cause A — `success_hold_seconds=1.0` is structurally unsatisfiable by these demos

`success_hold_seconds=1.0` compiles to **50 consecutive outer control steps**
(`success_hold_steps = 50`, `control_step_seconds = 0.02`). The stock demos
carry a mean of **20.3** steps after their first rewarding step; the longest
tail across all 60 demos is **26** steps (0.52 s) and 9 demos have none at all.
**0 of 60 demos are long enough to satisfy a 1.0 s hold**, even under a
physically perfect replay. This is why the perfect-substrate row collapses
0.750 → 0.000 when the only change is the hold.

### Cause B — the HOMIE base loses the pelvis

Pelvis tracking error of the raw replay (n = 30 demos, metres):

| era | at reset, before any action | mid-demo | end of demo | success |
|---|---|---|---|---|
| `floating` (demos' own `legacy_delta` base) | 0.0004 | 0.013 | 0.013 | 0.500 (trimmed demos) |
| `homie_cmd` (production) | **0.047** | 0.062 | **0.199** | 0.000 |

The HOMIE reset (knees-bent init at `init_pelvis_z=0.92` + 160 warmup steps)
already displaces the pelvis **4.7 cm** from where the demo starts, and
open-loop integration of the velocity commands grows that to **20 cm** by the
end of the demo — against 1.3 cm for the kinematic floating base. Grasping a
plate does not tolerate 20 cm. Raising `cmd_clip` from 1.0 to 5.0 (the demo
pelvis deltas divided by dt saturate the ±1 m/s clip) recovers only
0.000 → 0.100.

**Consequence for the open item**: a completed 100 K-frame CQN-AS run scoring
exactly 0.0 on all 40 evals is fully explained by the environment, without
invoking the algorithm. The demos it trains on cannot be executed in the env it
is evaluated in, by any controller, because (A) no demo is long enough to hold
success for 1 s and (B) the HOMIE base cannot follow the demo pelvis path
open-loop.

---

## 2. Step 2 — success vs `j`, full-episode open loop

Restore the episode's initial state, then execute the whole demo reprojected
chunk by chunk (consecutive, non-overlapping 16-step chunks) onto `j` knots,
open loop. n = 60 demos per cell.

### 2a. `floating` era, `success_hold_seconds=0` (the interpretable substrate)

| `j` | knots | success | mean reward | % of `j=16` | mean skeleton RMSE |
|---|---|---|---|---|---|
| 2 | 0, 15 | 0.467 | 0.467 | 62 % | 0.0905 |
| **3** | 0, 7, 15 | **0.817** | 0.817 | **109 %** | 0.0581 |
| 5 | 0, 3, 7, 11, 15 | 0.767 | 0.767 | 102 % | 0.0367 |
| 9 | 0,1,3,5,7,9,11,13,15 | 0.733 | 0.733 | 98 % | 0.0243 |
| 16 | all | 0.750 | 0.750 | 100 % | 0.0000 |

`j = 16` reproduces step 1's `floating_hold0` number **exactly** (0.750 vs
0.750, same 60 demos), so the projection/execution path has no bug.

Binomial standard error at n = 60, p ≈ 0.75 is 0.056. `j ∈ {3, 5, 9}` are all
within ~1.2 se of the control — statistically indistinguishable from it.
`j = 2` is 5.1 se below `j = 3` and 3.6 se below the control: a real drop.

### 2b. `homie` production era (for contrast)

| `j` | 2 | 3 | 5 | 9 | 16 |
|---|---|---|---|---|---|
| success | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Flat zero at every `j`, including the identity. Nothing in this era is
interpretable — which is the point of section 1.

---

## 3. Step 3 — per-chunk tracking error

The reference rollout (exact demo actions) is snapshotted at every chunk
boundary; for each `j` the sim state is restored there and the 16-step skeleton
is run open loop. Deviation is measured against the reference rollout's **own
achieved trajectory**, so `j = 16` is by construction the restore-error probe.
End-effector deviation = mean of the two wrist-site position errors, metres.
n = 426 chunks (40 demos), `floating` era.

| `j` | mean EE @ chunk end | p90 EE @ chunk end | mean EE max-over-chunk | mean joint err (rad) | p90 joint err |
|---|---|---|---|---|---|
| 2 | 0.0174 | 0.0383 | 0.0210 | 0.0785 | 0.1932 |
| 3 | 0.0056 | 0.0116 | 0.0066 | 0.0423 | 0.0918 |
| 5 | 0.0016 | 0.0034 | 0.0020 | 0.0157 | 0.0359 |
| 9 | 0.0006 | 0.0014 | 0.0009 | 0.0081 | 0.0118 |
| 16 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 |

Production `homie` era, n = 437 chunks (40 demos) — same shape, ~35 % smaller
because the HOMIE arm PD is stiffer:

| `j` | 2 | 3 | 5 | 9 | 16 |
|---|---|---|---|---|---|
| mean EE @ chunk end (m) | 0.0111 | 0.0042 | 0.0013 | 0.0005 | 0.0000 |

### 3b. Deviation by episode phase (`floating`, mean EE at chunk end, m)

| `j` | early third | mid third | late third |
|---|---|---|---|
| 2 | 0.0174 | **0.0292** | 0.0077 |
| 3 | 0.0060 | 0.0088 | 0.0027 |
| 5 | 0.0018 | 0.0023 | 0.0009 |
| 9 | 0.0006 | 0.0009 | 0.0004 |

Error peaks in the **mid** phase (reach and transport, the fastest arm motion)
and is smallest **late** (placement, where contact happens). There is no
contact-phase blow-up, which is why this is PASS and not GRADED.

### 3c. Restore verification (required before trusting the above)

`j = 16` after a restore deviates from the reference by **0.0000 m** on average
and **1.0e-4 m** worst case over 426 restores. The channel-wise probe
(`restore_check.py`, full-model state rather than the observation vector):

| era | chunks | max abs qpos residual | max abs qvel residual | max hand residual (m) | chunks exactly 0 |
|---|---|---|---|---|---|
| `floating` | 170 | 5.0e-4 | 0.156 | **3.4e-5** | 19.4 % |
| `homie` | 117 | 1.5e-2 | 1.77 | **5.7e-5** | 18.8 % |

**The restore is not bit-exact on this scene.** The GOLDEN record in
`tests/smoke/snapshot_roundtrip_probe.py` (`max|A-B| = 0.000e+00`) was measured
on a *walking* sequence with no object manipulation; `move_plate` adds a
contact-rich plate/counter interaction and the residual is real. It is however
tiny where it matters: the replayed wrist trajectory reproduces to
**≤ 5.7e-5 m (0.06 mm)** worst case, and the residual sits mostly in the
robot's own leg joints (argmax of |Δqpos| lands in the object's free joint in
only 6 of 95 sampled chunks). The `j = 2` signal (17.4 mm) is **300×** that
noise floor and `j = 9` (0.6 mm) is still **10×** above it, so every row of
table 3 is above the floor. Do not push this harness below `j = 9` resolution
without re-deriving the floor.

---

## 4. Step 4 — controls

### 4a. Zero-order hold instead of linear interpolation (`floating`, n = 60)

| `j` | 2 | 3 | 5 | 9 | 16 |
|---|---|---|---|---|---|
| linear | 0.467 | 0.817 | 0.767 | 0.733 | 0.750 |
| **zero-order hold** | **0.117** | **0.583** | 0.817 | 0.717 | 0.750 |

Interpolation is what buys the coarse end. At `j = 2` it is worth 0.35 success
(0.117 → 0.467) and at `j = 3` 0.23 (0.583 → 0.817); at `j ≥ 5` the two are
indistinguishable. **The skeleton's value at low `j` comes from the linear
segments, not merely from picking good knots** — a ZoomQ variant that executed
held actions between knots would lose the anytime property that this gate just
confirmed.

### 4b. Knot perturbation (`floating`, n = 60)

The middle knot of every chunk is displaced by one quantum before projection.
`fine_bin` = 2/5³ = **0.016** (the finest CQN-AS bin over [-1, 1] with the live
runs' `levels=3, bins=5`); `window` = **0.15** (ZoomQ's round-1 residual-window
half-width). `plus`/`minus` shift every action dimension the same way;
`rand` rounds each dimension independently — the realistic quantiser model.

| perturbation | `j = 3` | `j = 5` |
|---|---|---|
| none | 0.817 | 0.767 |
| ±1 fine bin, random sign per dim | **0.700** | **0.733** |
| +1 fine bin, all dims | 0.033 | 0.033 |
| −1 fine bin, all dims | 0.550 | 0.867 |
| ±1 window, random sign per dim | 0.083 | 0.233 |
| ±1 window, all dims same sign | 0.000 | 0.000 |

Realistic quantiser noise costs 0.12 (j=3) / 0.03 (j=5) — survivable. One
*window* quantum destroys the task at every `j`. **A knot must be resolved to
well inside its residual window; committing at window resolution is not
executable.** The `plus` row also shows the skeleton is far more sensitive to a
correlated DC bias across action dimensions (0.033) than to independent
rounding of the same magnitude (0.700) — a systematic offset in the decoder is
much worse than its nominal bin width suggests.

---

## What would change this conclusion

1. **The substrate.** The PASS is measured in the `floating` era, where the
   pelvis is a kinematic floating base driven by the demo's own base slots. The
   production ZoomQ env drives the pelvis through the HOMIE `mjlab_h1` policy,
   and section 1 shows that env cannot execute these demos at all. If ZoomQ's
   anytime claim has to hold *in the production env*, this gate has not tested
   it — it has tested the arm/gripper trajectory in isolation from the
   locomotion controller. Re-running section 2 on native HOMIE demos (recorded
   with `base_action_mode=homie_cmd` and a ≥ 1 s success tail) would settle it.
   That is the only experiment that could turn PASS into VETO.
2. **A longer horizon.** These chunks are 16 outer steps = 0.32 s. A slower
   control rate or a longer chunk would put more curvature inside one chunk and
   could move the depth floor above 3.
3. **One task.** `move_plate` is a reach-grasp-transport-place with no fine
   in-contact regulation. A task whose success depends on a short high-frequency
   contact event (`flip_cup`, `flip_cutlery`) could show the GRADED pattern this
   one does not. The floor found here is a floor for `move_plate`, not a
   universal one.
4. **Open loop is generous in one direction and harsh in the other.** Open-loop
   replay has no feedback to correct a projection error (harsh), but it also
   never has to *choose* the knots — a learned policy must predict them, and
   section 4b shows a one-window error at a single knot is fatal. Gate 1 bounds
   the executability of a skeleton; it does not bound the learnability of one.
5. **Demos, not policy.** Every trajectory here is an expert demonstration.
   Skeletons of a policy's own (noisier, less smooth) chunks may project worse;
   Gate 0's residual statistics are the right cross-check.

---

## Reproduce

```bash
cd /mnt/workspace/zoomq/gates
source /usr/local/PPU_SDK/envsetup.sh
PY=/mnt/workspace/anchorq/.venv/bin/python

# step 1, era localisation table
MUJOCO_GL=egl $PY gate1_replay.py --mode raw --era-sweep --num-demos 60 \
    --workers 10 --out step1_era_sweep.json
# step 2
MUJOCO_GL=egl $PY gate1_replay.py --mode full --era floating --num-demos 60 \
    --workers 10 --j 2,3,5,9,16 --out step2_full_floating.json
MUJOCO_GL=egl $PY gate1_replay.py --mode full --era homie    --num-demos 60 \
    --workers 10 --j 2,3,5,9,16 --out step2_full_homie.json
# step 3
MUJOCO_GL=egl $PY gate1_replay.py --mode chunk --era floating --num-demos 40 \
    --workers 10 --j 2,3,5,9,16 --out step3_chunk_floating.json
# step 4
MUJOCO_GL=egl $PY gate1_replay.py --mode zoh     --era floating --num-demos 60 \
    --workers 10 --j 2,3,5,9,16 --out step4_zoh_floating.json
MUJOCO_GL=egl $PY gate1_replay.py --mode perturb --era floating --num-demos 60 \
    --workers 10 --j 3,5 --out step4_perturb_floating.json
```

Total wall clock for the whole gate: ~6 minutes on 10 workers. Deterministic:
each demo replays from its own recorded `demo.seed`, demos are ordered by that
seed (`DemoStore` returns them shuffled), and the only other RNG is the seeded
`rand` sign draw in step 4b.
