# Gate 1 follow-up — per-demo replay verdicts, and is the 0.750 real?

**Date**: 2026-08-16. **Substrate**: BiGym 1 semantics, kinematic **floating**
pelvis only (`lowerbody_policy.enabled=false`, `base_action_mode=legacy_delta`,
`support_base_with_legs=false`, `success_hold_seconds=0`). The `homie` era is
not touched here.
**Tasks**: `move_plate` (60 stock demos), `drawer_top_close` (51 stock demos).
**Numbers**: `gates/replay_verdicts.json`, `gates/gate1_frequency_sweep.json`,
`gates/gate1_meta_probe.json`, `gates/gate1_raw/offset_probe.json`.
**Harness**: `gates/gate1_replay.py` (extended, all old entry points intact) +
`gate1_verdicts.py`, `gate1_freq_sweep.py`, `gate1_prebuild_rates.py`,
`gate1_meta_probe.py`, `gate1_offset_probe.py`, `gate1_merge_sweep.py`,
`gate1_tables.py`.

---

## Headline

1. **The 0.750 is neither a physics ceiling nor a rate artefact — it is a
   harness artefact.** `gate1_replay.py` executed `actions[1:]`, skipping the
   demo's first recorded action on the belief that index 0 is a dummy reset
   action. The cached `DemoStore` demos carry no such dummy. Executing every
   recorded action gives **0.867** on `move_plate` at the configured 50 Hz,
   against 0.750 for the truncated stream. Same demos, same seeds, same env.
2. **Control rate is not the explanation for anything.** Once the dropped
   action is restored, `move_plate` replay success is **flat at 0.867–0.900
   across 20 Hz → 500 Hz** (a 25× span, 11 rates): span 0.033, i.e. inside
   1 se. Under the old convention the same sweep spans 0.283 (0.583 → 0.867) —
   that entire apparent rate curve was the missing action, whose cost scales
   with the duration of one control step.
3. **The remaining `move_plate` drop is 8/60 = 0.133, and all 8 are accounted
   for**: 7 demos whose own stored recording never reaches a reward at all, and
   1 demo (`d4701200`, index 46) whose recording first rewards at outer step 308
   while `episode_length=3000 / dsr=10` caps the episode at 300. **Zero demos
   fail for a physics or timing reason.**
4. **`drawer_top_close` drops nothing: 51/51 = 1.000, at every one of the 11
   rates, under both conventions.**
5. **A positional demo index is NOT a stable key** — `DemoStore._get_demos()`
   ends in `np.random.shuffle(files)`. `replay_verdicts.json` is keyed on
   `uuid`. Read §1 before consuming it.
6. **MuJoCo version drift is real but small and in our favour**: rebuilding the
   50 Hz demo cache under MuJoCo 3.11.0 instead of the shipped 3.1.5 leaves the
   action tensors **bit-identical** (60/60 and 51/51, max |Δ| = 0.0) and moves
   `move_plate`'s recorded success from 0.850 to 0.900.

---

## 1. The demo key (read this before using `replay_verdicts.json`)

**The position in the list `DemoStore` returns is not a stable key and cannot be
used to identify a demonstration.**

`demonstrations/demo_store.py::DemoStore._get_demos` ends with
`np.random.shuffle(files)`, so the order `bigym/loco/env.py::BiGym.get_demos`
hands to `train_cqn_as_bigym.py::load_rlbench_demos` depends on the **global
numpy RNG state at load time**. `train_cqn_as_bigym.py:63` calls
`utils.set_seed_everywhere(cfg.seed)` (which does `np.random.seed(seed)`) and
the demos are loaded much later, at line 1538, after the env constructors have
themselves drawn from `np.random` (`BiGymEnv.__init__` draws
`np.random.randint(2**32)` for `start_seed`). Measured directly
(`gate1_meta_probe.json`):

| probe | result |
|---|---|
| same numpy seed, two `get_demos` calls | **identical order** |
| numpy seed 0 vs numpy seed 1 | **different order** |
| returned order == sorted-by-uuid order | no |

So the load position is reproducible for one `(cfg.seed, call sequence)` and
changes with either. It is not a demo identity.

**What is stable.** Every demo carries, in its `.safetensors` metadata, a
`uuid` (which is also its filename, `Metadata.filename = f"{uuid}.safetensors"`)
and a `seed` (its env reset seed). Both are unique across the demo set for both
tasks — 60/60 unique uuids and 60/60 unique seeds on `move_plate`, 51/51 and
51/51 on `drawer_top_close`.

`replay_verdicts.json` therefore reports, per demo, `uuid`, `seed`, and an
`index` defined as **the rank of the demo in `sorted(uuid)` order** —
deterministic, and derivable from a directory listing without loading anything.
`success_indices` / `fail_indices` use that index; `success_uuids` /
`fail_uuids` / `success_seeds` / `fail_seeds` are also present.

> **A consumer must join on `uuid` (or `seed`). Joining on a positional index
> into the trainer's loaded list will silently mislabel demos.**
> To actually use this in the trainer, `BiGym.get_demos` has to be taught to
> keep `raw_demo.metadata.uuid` alongside the converted timesteps — the
> conversion currently discards it, so there is no way to filter downstream of
> `get_demos` as it stands today.

---

## 2. Task 1 — per-demo replay verdicts

Replay every demo action-for-action from its own recorded reset seed
(`env.reset(seed=demo.seed)`), floating era, `success_hold_seconds=0`, untrimmed,
at the configured `demo_down_sample_rate` from
`cfgs/bigym_task/<task>.yaml` (10 for both). Success = `episode_reward >= 0.25`
(the repo's own eval predicate).

| task | dsr | Hz | ms/step | n | **replay success (all actions)** | **drop fraction** | Gate-1 convention (drops action 0) |
|---|---|---|---|---|---|---|---|
| `move_plate` | 10 | 50 | 20 | 60 | **52/60 = 0.867** | **0.133** | 45/60 = 0.750 |
| `drawer_top_close` | 10 | 50 | 20 | 51 | **51/51 = 1.000** | **0.000** | 51/51 = 1.000 |

`enable_all_floating_dof` differs between the tasks (`false` for `move_plate`,
`true` for `drawer_top_close`) and it changes the floating-DOF set and hence the
demo directory (`..._pelvis_x_pelvis_y_pelvis_rz_...` vs
`..._pelvis_x_pelvis_y_pelvis_z_pelvis_rz_...`), so it is read from the task
yaml rather than hard-coded.

### 2a. `move_plate` — the 8 dropped demos

| index | uuid | seed | demo steps | first rewarding step **in the recording** | recorded reward | fails at N of 11 rates |
|---|---|---|---|---|---|---|
| 0 | `09431449…` | 783198766 | 226 | never | 0.0 | 10 |
| 13 | `57a29e4b…` | 4089832376 | 184 | never | 0.0 | 11 |
| 21 | `765f42df…` | 3129168941 | 182 | never | 0.0 | 11 |
| 25 | `7c29f4e4…` | 1647355863 | 171 | never | 0.0 | 8 |
| 26 | `7cfa15a5…` | 940041192 | 163 | never | 0.0 | 9 |
| 29 | `8beece23…` | 666157710 | 187 | never | 0.0 | 11 |
| 46 | `d4701200…` | 143646756 | **334** | **308** | 25.0 | 11 |
| 59 | `fa2c53e0…` | 1935868678 | 207 | never | 0.0 | 1 |

**Seven of the eight are demos that never reach a reward in their own stored
recording.** They are not replay failures — they are demonstrations of a
failure. Nine `move_plate` demos are in that state; the replay actually
*rescues* two of them (indices 10 and 22 succeed under MuJoCo 3.11.0 despite a
zero recorded reward from the 3.1.5-era cache).

**The eighth, index 46, is a time-limit truncation, not a physics failure.** Its
recording first rewards at outer step 308; the loco env truncates at
`episode_length // demo_down_sample_rate = 3000 // 10 = 300`. Both numbers scale
as 1/dsr, which is why this demo is the one demo that fails at all 11 rates.
Raising `episode_length` to ≥3340 recovers it (`norm_all_nolimit` in
`gate1_raw/offset_probe.json`: 0.867 → 0.883).

**Of the 51 demos whose recording did reach a reward, exactly 1 fails on replay
(index 46, the truncation).** The physics itself loses nothing.

### 2b. Systematic or random?

| covariate | mean over the 52 successes | mean over the 8 failures | point-biserial r with success |
|---|---|---|---|
| demo length (outer steps) | 179.5 | 206.8 | −0.312 |
| first rewarding step in the recording (−1 = never) | 148.2 | 37.6 | +0.595 |
| recorded reward of the demo | 22.27 | 3.13 | +0.695 |
| fraction of action entries clipped at ±1 | 0.000 | 0.000 | n/a |

**Systematic, and on exactly one axis: whether the demonstration ever succeeded
in the first place.** `recorded reward` (r = +0.695) and `first rewarding step`
(r = +0.595) are both restatements of "did this demo reach the goal"; failures
being ~15 % longer (r = −0.312) is the same fact seen from the other side (a
demo that never succeeds runs to its natural end).

**Not clustered in initial conditions.** Only **2** of the 37 scene `qpos`
coordinates vary at all across the 60 reset seeds — `qpos[30]` and `qpos[31]`,
the plate free joint's x and y (std 0.030 m and 0.083 m). The strongest
correlation between either of them and replay success is **|r| = 0.150**
(x: mean 0.7029 over successes vs 0.7160 over failures; y: 0.3071 vs 0.3277).
There is no initial-condition story here.

`drawer_top_close` has **zero** varying `qpos` coordinates — its reset is fully
deterministic — and zero failures, so nothing to analyse.

---

## 3. Task 2 — control frequency

### 3.1 Compiled MuJoCo timestep and substeps per control step

Read off the compiled model at runtime (`model.opt.timestep`), not from a yaml —
`gate1_meta_probe.py` → `gate1_meta_probe.json`.

| task | scene actually loaded | compiled `opt.timestep` | nominal `PHYSICS_DT` | substeps at dsr=10 | `control_step_seconds` |
|---|---|---|---|---|---|
| `move_plate` | `/mnt/data/bigym/bigym/envs/xmls/world.xml` | **0.002 s** | 0.002 s | **10** | **0.020 s** |
| `drawer_top_close` | `/mnt/data/bigym/bigym/envs/xmls/world.xml` | **0.002 s** | 0.002 s | **10** | **0.020 s** |

Checked at dsr = 5, 10 and 25 for both tasks: compiled timestep is 0.002 s in
every case, substeps = dsr exactly, and `control_step_seconds = 0.002 × dsr`.
`BiGymEnv.__init__` prints
`"[bigym] compiled timestep …s != nominal …s: control substeps X -> Y"`
whenever a robot XML has smuggled an `<option timestep>` into the scene. That
line appears **0 times** across every run in this follow-up.

> **The `CLAUDE.md` claim "H1 = 10×2 ms, unchanged" is re-verified on this
> substrate.** The G1-dex3 `timestep=0.001` merge that produced the 2026-07-16
> era boundary does not touch the H1 `world.xml` scene these two tasks load.
> Solver 2 / integrator 3 on both; `nq/nv/nu` = 37/36/15 for `move_plate`,
> 38/38/16 for `drawer_top_close` (the extra DOF is `pelvis_z`).

### 3.2 `control_frequency` and `demo_down_sample_rate`

**There is exactly one rate knob, and `control_frequency` is not separately
settable.** `bigym/loco/env.py` builds the env with

```
control_frequency = CONTROL_FREQUENCY_MAX // demo_down_sample_rate      # 500 // dsr
```

and `BiGym.get_demos` asks `DemoStore` for demos decimated to **the same**
frequency. The demo rate and the control rate are the same number by
construction; they cannot diverge, which is precisely why a
recording-vs-replay rate mismatch cannot arise on this path. Legal range:
`CONTROL_FREQUENCY_MIN=20 ≤ 500//dsr ≤ CONTROL_FREQUENCY_MAX=500`, i.e.
`dsr ∈ [1, 25]`.

| dsr | control Hz | s / control step | substeps | max outer steps (`episode_length=3000`) |
|---|---|---|---|---|
| 1 | 500 | 0.002 | 1 | 3000 |
| 5 | 100 | 0.010 | 5 | 600 |
| **10 (configured)** | **50** | **0.020** | **10** | **300** |
| 20 | 25 | 0.040 | 20 | 150 |
| 25 | 20 | 0.050 | 25 | 120 |

### 3.3 What do the demos record about their own rate?

**Nothing. The `.safetensors` carry no rate field at all.** The complete
metadata field set is

```
seed, uuid, date, observation_mode, package_versions, environment_data
```

and `environment_data` is `env_name, action_mode_name, action_mode_absolute,
floating_base, floating_dofs, observation_config, reset_positions, robot_name`.
A substring scan of the whole serialised metadata for
`control_step_seconds | frequency | hz | timestep | dt | rate | decimat`
returns **zero hits** on both tasks. There is no `control_step_seconds` field
on stock 0.9.0 demos.

**But the question is still answerable, from the loader rather than the file.**
`DemoConverter.decimate` refuses to run unless the source is at
`CONTROL_FREQUENCY_MAX`:

```python
if original_freq != CONTROL_FREQUENCY_MAX:
    raise RuntimeError("Demonstrations with frequency != 500 can't be decimated.")
```

so every shipped demo is a **500 Hz** recording by construction, and every rate
you ever replay at is produced *from that 500 Hz original* by
`decimate(demo, 500//dsr, 500)` followed by `create_demo_in_new_env`. The rate
is carried by the cache directory name (`…/state/{freq}hz/`), not by the file.
**A recording/replay rate mismatch is structurally impossible here** — which is
the answer to "can we even check this".

What the demos *do* record is the package set they were made under:
`{"mujoco": "3.1.5", "bigym": "4.1.0"}`, identically on **60/60** `move_plate`
and **51/51** `drawer_top_close` demos, against an installed MuJoCo **3.11.0** —
the drift Gate 1 blamed. Section 3.6 measures it.

### 3.4 The sweep — `move_plate` (n = 60 per cell)

Same open-loop raw replay as Gate 1, floating era, hold 0, untrimmed. Every
non-50 Hz cache was regenerated from the 500 Hz originals into a **private**
cache root under `gates/` (`build_private_demo_cache.sh` + symlinks), so
`/mnt/data/bigym-cache` was never written to.

| dsr | control Hz | ms / control step | **all actions (correct)** | Gate-1 convention (drops action 0) | demo's own recorded success |
|---|---|---|---|---|---|
| 1 | 500 | 2 | **0.883** (53/60) | 0.867 | 0.917 |
| 2 | 250 | 4 | **0.883** (53/60) | 0.867 | 0.900 |
| 3 | 166 | 6 | **0.900** (54/60) | 0.833 | 0.917 |
| 4 | 125 | 8 | **0.883** (53/60) | 0.867 | 0.900 |
| 5 | 100 | 10 | **0.867** (52/60) | 0.867 | 0.883 |
| 8 | 62 | 16 | **0.883** (53/60) | 0.750 | 0.900 |
| **10 (configured)** | **50** | **20** | **0.867** (52/60) | **0.750** | 0.850 |
| 12 | 41 | 24 | **0.883** (53/60) | 0.650 | 0.900 |
| 15 | 33 | 30 | **0.883** (53/60) | 0.700 | 0.900 |
| 20 | 25 | 40 | **0.883** (53/60) | 0.617 | 0.900 |
| 25 | 20 | 50 | **0.900** (54/60) | 0.583 | 0.917 |

Binomial 1 se at n = 60, p ≈ 0.88 is 0.041.

* **All actions: span 0.033 over a 25× range of control rate** — 0.8 se,
  statistically indistinguishable from flat. No rate does materially better and
  none does materially worse. Nothing here reaches 0.9+ in a way the configured
  rate does not; the two cells that read 0.900 are two demos above the
  configured cell, well inside noise.
* **Gate-1 convention: span 0.283**, monotone in the control step length
  (0.867 at 2–10 ms → 0.583 at 50 ms). That curve is entirely the missing first
  action: one skipped action costs 2 ms of motion at 500 Hz and 50 ms at 20 Hz.
* The demos' own recorded success (the reward stored when the cache was built,
  i.e. a bare-`BiGymEnv` replay of the raw actions) is likewise **flat at
  0.883–0.917** across all rates. The one low cell, 0.850, is the shipped
  50 Hz cache — which is the only cell built under MuJoCo 3.1.5.

### 3.5 The sweep — `drawer_top_close` (n = 51 per cell)

| dsr | 1 | 2 | 3 | 4 | 5 | 8 | **10** | 12 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control Hz | 500 | 250 | 166 | 125 | 100 | 62 | **50** | 41 | 33 | 25 | 20 |
| all actions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| Gate-1 convention | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| demo's own recorded success | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |

51/51 everywhere. `drawer_top_close` is a clean demo set and is insensitive to
the first-action bug (its trajectories have enough slack; ~9.7 % of its action
entries are clipped at ±1 by the training normalisation at every rate, and even
that does not cost it a single episode).

### 3.6 Control — is the sweep confounded by MuJoCo drift?

The shipped 50 Hz cache was built under MuJoCo 3.1.5; every other rate had to be
regenerated now, under 3.11.0. To rule that out, the 50 Hz set was **also**
regenerated under 3.11.0 into a second private cache
(`gates/bigym_cache_regen50`) and compared demo-by-demo:

| | `move_plate` | `drawer_top_close` |
|---|---|---|
| demos with a **bit-identical** action tensor | **60/60** | **51/51** |
| max abs difference of Σ\|action\| per demo | **0.0** | **0.0** |
| demo lengths | identical | identical |
| clipped fraction | identical (0.000) | identical (0.0972) |
| recorded success at build time | 0.850 (3.1.5) → **0.900** (3.11.0) | 1.000 → 1.000 |

**The action stream a demo carries does not depend on the MuJoCo version used to
build the cache** — decimation is deterministic arithmetic on the 500 Hz
`executed_action`s. So the sweep is not confounded.

The physics version does move the *outcome*, on `move_plate` only, and it moves
it **up**: 51/60 → 54/60 recorded successes. MuJoCo drift is real here but it is
worth ~+0.05, not −0.25, and it is not the reason any demo is dropped.

### 3.7 Task 2.5 — same demos, or different ones?

Failure-set overlap across the 11 rates (mean pairwise Jaccard; the "chance"
column is the expected Jaccard of two independent sets of the observed sizes):

| convention | mean Jaccard | chance | 0 failures at any rate | fails at **every** rate | fails at ≥1 rate |
|---|---|---|---|---|---|
| all actions (correct) | **0.758** | 0.062 | 48/60 | 4 | 12 |
| Gate-1 convention | 0.417 | 0.124 | 24/60 | 1 | 36 |

**Same demos.** With the harness corrected, 48 of 60 `move_plate` demos never
fail at any rate, and the observed overlap is 12× the independence baseline.
The distribution of "how many of the 11 rates does this demo fail at" is
bimodal: 4 demos fail at all 11 rates (indices 13, 21, 29, 46), 3 more fail at
8–10 of them (indices 0, 25, 26), and 5 are near-misses that fail at only one or
two rates (indices 34, 41, 45, 59 at one; index 10 at two) — the borderline
demos that a nudge in contact timing pushes either way.

Under the old convention the picture was the opposite — 36 of 60 demos fail
somewhere, mean Jaccard 0.417, a broad histogram — which would have read as a
timing-sensitivity story. **That reading was an artefact of the harness, and it
is worth noting that the artefact produced exactly the signature one would have
cited as evidence for the rate hypothesis.**

### 3.8 Where the first-action bug came from, and its size

`gate1_replay.py::load_demos` did

```python
# index 0 is the reset step's dummy action ... so the executable sequence is [1:]
"actions": np.clip(acts[1:], -1, 1)
```

The cached demos have no dummy reset step: `DemoStore.get_demos` builds them
with `DemoConverter.create_demo_in_new_env`, which does `env.reset(seed=…)` and
*then* steps with **every** source action, so its timestep 0 is already a
post-action state carrying a real executed action. `gate1_offset_probe.py`
separates the three differences between the harness and the demo's own recorded
outcome, `move_plate`, dsr = 10:

| variant | success |
|---|---|
| `norm_drop1_limit` — Gate 1's convention | **0.750** |
| `norm_all_limit` — execute action 0 too | **0.867** |
| `norm_all_nolimit` — also lift the 300-outer-step time limit | **0.883** |
| the demo file's own stored reward (bare env, raw float64 actions, no limit) | 0.850 |
| same, cache rebuilt under MuJoCo 3.11.0 | 0.900 |

So on `move_plate` at 50 Hz: **first action = 0.117 (7 demos), episode-length
truncation = 0.017 (1 demo)**, and the residual between 0.883 and the bare-env
0.900 is the training normalisation round-trip plus the loco wrapper (1 demo).
`drawer_top_close` is 1.000 under all five variants.

The default in `gate1_replay.py` is left at drop-first so the published Gate-1
tables still reproduce byte-for-byte (verified: `--mode raw --era floating
--num-demos 60` still returns exactly 0.750). `--keep-first` opts into the
correct stream, and it is the default in `gate1_verdicts.py` /
`gate1_freq_sweep.py`.

> **This invalidates the Gate-1 report's section 2/3/4 numbers as absolute
> values**, since every one of them replays `actions[1:]`. It probably does not
> invalidate the *comparisons* — the `j`-skeleton rows all share the same
> offset, and the `j = 16` control was 0.750 for the same reason the raw replay
> was — but §2's "`j = 3` reaches 109 % of the control" is a ratio against a
> depressed control and should be re-measured with `--keep-first` before it is
> quoted again. Not re-run tonight; flagged.

---

## 4. Plain answers

**Q. What fraction is dropped per task?**
`move_plate` **0.133** (8 of 60). `drawer_top_close` **0.000** (0 of 51).
Under the harness convention Gate 1 used, `move_plate` would read 0.250 (15 of
60) and `drawer_top_close` 0.000.

**Q. Are the failures systematic or random?**
Systematic, on one axis only: **7 of the 8 `move_plate` drops are demos whose
own stored recording never reaches a reward** (r = +0.695 between recorded
reward and replay success). The 8th is an `episode_length` truncation. They are
**not** clustered in initial conditions (only 2 scene coordinates vary at all —
the plate x/y — and |r| ≤ 0.15), and they are not clustered in episode length
except as a consequence of never succeeding.

**Q. What is the compiled MuJoCo timestep, and how many substeps per control
step?**
**0.002 s** on both tasks — equal to the nominal `PHYSICS_DT`, with no XML
override anywhere (the "compiled timestep != nominal" warning never fires).
Substeps = `demo_down_sample_rate` exactly; **10** at the configured rate.

**Q. What `control_frequency` / `demo_down_sample_rate` are in effect and what
seconds-per-control-step do they imply?**
`demo_down_sample_rate = 10` → `control_frequency = 500 // 10 = 50 Hz` →
`control_step_seconds = 0.002 × 10 = 0.020 s`. They are the same knob;
`control_frequency` is not independently settable through `bigym_env.make`.

**Q. Do the demos record their recording rate?**
No. The stock `.safetensors` carry `seed, uuid, date, observation_mode,
package_versions, environment_data` and nothing rate-shaped (zero substring
hits). But the question is moot: `DemoConverter.decimate` hard-refuses any
source that is not 500 Hz, so every shipped demo is a 500 Hz recording and every
replay rate is derived from it — a recording/replay rate mismatch cannot happen
on this path. What they *do* record is `mujoco 3.1.5 / bigym 4.1.0`, uniformly.

**Q. Is 0.750 the ceiling, or does some other rate do materially better?**
**Neither.** 0.750 is a bug in the replay harness. No rate does materially
better than any other: with the demo's full action stream, `move_plate` sits at
0.867–0.900 from 20 Hz to 500 Hz, span 0.033 = 0.8 se. The real ceiling at the
configured rate is **0.867 (52/60)**, it is reached at every rate, and the
remaining 8 demos are 7 broken demonstrations plus 1 `episode_length`
truncation. Raising `episode_length` from 3000 to ≥3340 takes it to 0.883
(53/60); the last 7 cannot be recovered by any rate or any timestep because
they never demonstrate the task.

**Q. Same demos across rates, or different ones?**
**Same.** Mean pairwise Jaccard of the failing sets is 0.758 against a chance
baseline of 0.062; 48 of 60 demos never fail at any rate. The "different demos
at different rates" appearance (Jaccard 0.417, 36 of 60 failing somewhere) is
produced by the first-action bug and disappears with it.

---

## 5. What this means for the plan

* **`drawer_top_close` needs no filtering.** 51/51 replay. Admitting only
  replay-successful demos is a no-op there.
* **`move_plate` should drop 8 of 60**, and 7 of those are demos that never
  demonstrate the task — exactly the "mislabeled supervision" the plan is aimed
  at. `replay_verdicts.json.move_plate.fail_uuids` is the list.
  The 8th (`d4701200…`, index 46) is a *good* demo being cut by
  `episode_length`; either raise `episode_length` to ≥3340 and keep it, or drop
  it knowing why.
* **Filtering has to be keyed on `uuid`, and the trainer cannot do that today.**
  `BiGym.get_demos` returns converted timestep lists with the uuid discarded;
  adding the filter means threading `raw_demo.metadata.uuid` (or `.seed`)
  through `get_demos` before `convert_demo_to_timesteps` drops it.
* **Do not spend more time on the control rate.** It is one knob, it is
  self-consistent by construction, the H1 scene has no smuggled timestep, and
  the sweep is flat. The 2026-07-16 G1 era boundary genuinely does not apply to
  H1.
* **Re-derive the Gate-1 §2 skeleton table with `--keep-first` before quoting
  its ratios.** Its `j = 16` control is the 0.750 artefact.

---

## 6. Reproduce

```bash
ssh ali
source /usr/local/PPU_SDK/envsetup.sh
export MUJOCO_GL=egl
PY=/mnt/workspace/anchorq/.venv/bin/python
cd /mnt/workspace/zoomq/gates

# 0. private demo cache (never write to /mnt/data/bigym-cache)
bash build_private_demo_cache.sh
for T in move_plate drawer_top_close; do for D in 1 2 3 4 5 8 12 15 20 25; do
  $PY gate1_prebuild_rates.py $T $D $PWD/bigym_cache > gate1_raw/prebuild/${T}_dsr${D}.json &
done; done; wait

# 1. compiled timestep / demo metadata / DemoStore ordering
$PY gate1_meta_probe.py > gate1_meta_probe.json

# 2. per-demo verdicts (both conventions)
$PY gate1_verdicts.py --workers 26 --out replay_verdicts.json

# 3. the rate sweep, both conventions, then merge
$PY gate1_freq_sweep.py --rates 1,2,3,4,5,8,10,12,15,20,25 --workers 26 \
    --out gate1_raw/sweep_dropfirst.json
$PY gate1_freq_sweep.py --rates 1,2,3,4,5,8,10,12,15,20,25 --workers 26 \
    --keep-first --out gate1_raw/sweep_keepfirst.json
$PY gate1_merge_sweep.py

# 4. what the 0.750 actually was
$PY gate1_offset_probe.py 24

# 5. tables
$PY gate1_tables.py
```

Wall clock: ~4 min for the prebuilds, ~5 min per sweep, ~1 min for everything
else, on 26 workers of a contended 184-core box. Fully deterministic — every
demo replays from its own recorded reset seed and nothing draws from an RNG.
The old Gate-1 entry points are unchanged and still reproduce their published
numbers.
