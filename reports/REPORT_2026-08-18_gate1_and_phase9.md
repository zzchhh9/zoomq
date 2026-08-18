# ZoomQ — corrected Gate 1, the coarse-round deficit, and phase 9's 2×2

**2026-08-18 · move_plate · BiGym 1 floating base**

Everything below is measured. Demo-replay numbers come from `gates/gate1_replay.py`
at n=60 with no neural network involved; training numbers are successful training
rollouts read from `runs/*/train.csv` (`episode_reward`, which is exactly 1.0 on a
success), never from offline evals.

---

## 0. What changed, in one paragraph

The published Gate-1 table for move_plate was measured under an off-by-one replay
and is wrong. Re-measured at the deployment phase, a round-0 skeleton caps the task
at **0.100** against an unprojected **0.867** — and the entire loss is the first
~5 control steps after reset, not knot precision. That deficit is a property of
committing 16 steps open loop: replanning every 2 steps restores **0.833**.
Separately, ZoomQ is **not** broken — every arm's first success lands at episode
20-21 — but it is still 3-5× slower than the baseline, and its stopping rule is
provably inert on every arm ever run, including the ones carrying the fix that was
supposed to wake it up.

---

## 1. The corrected Gate 1

The old move_plate table (j=2 0.467, j=3 0.817, j=16 0.750) was measured with
`keep_first=False` — an off-by-one that drops the demo's first action. That shifts
the whole action array one step, and `project_sequence` cuts chunks at
`range(0, len, 16)`, so the shift moves every chunk boundary.

Phase 0 is the deployment phase: `train_cqn_as_bigym.py:716` replans only when
`episode_step % action_sequence == 0`, so with `temporal_ensemble=false` the
boundaries are exactly t = 0, 16, 32, …

| round r | knots j | SR | % of j=16 | skeleton RMSE |
|---|---|---|---|---|
| 0 | 2 | **0.100** | 12% | 0.0962 |
| 1 | 3 | **0.367** | 42% | 0.0619 |
| 2 | 5 | 0.833 | 96% | 0.0399 |
| 3 | 9 | 0.900 | 104% | 0.0241 |
| 4 | 16 | 0.867 | 100% | 0.0000 |

*n=60, floating era, hold=0, keep_first, phase 0.*

Gate 1 passes — **but only from round 2 up**. Rounds 0 and 1, which is exactly
where the anytime and latency claims live, sit at 12% and 42% of ceiling. ZoomQ
executes round 0 ~89% of the time (`depth_share_0` 0.884-0.892), so its structural
ceiling on this task is 0.100 before any learning question is asked.

## 2. It is not knot precision — the error budget

Gate 1 asks for an error-budget arm and it had never been run. `--knot-noise`
perturbs **every** knot by N(0, σ) before projecting:

| σ | j=2 | j=3 | j=16 |
|---|---|---|---|
| 0.000 | 0.100 | 0.300 | 0.800 |
| 0.008 — the CQN-AS baseline's measured knot MAE | 0.100 | 0.350 | 0.800 |
| **0.049 — ZoomQ's measured knot MAE** | 0.050 | 0.200 | **0.600** |
| 0.070 | 0.000 | 0.100 | 0.550 |
| 0.100 | 0.000 | 0.100 | 0.550 |

*n=20.* At ZoomQ's own knot error the full 16-knot chunk still scores 0.600.
Coarseness costs 2.5-3.5× more than knot error, so "the critic picks imprecise
knots" is refuted as the binding constraint.

## 3. The cause: the first five control steps

**Phase sweep** (`--phase` slides the chunk grid without changing which actions
execute or how many), j=3, n=60:

```
phase  0     1     2     3     4     5  |  6     7     8     9    10    11    12    13    14    15
SR   0.367 0.783 0.850 0.883 0.833 0.800| 0.567 0.467 0.350 0.383 0.383 0.317 0.367 0.400 0.383 0.367
RMSE 0.062 0.058 0.061 0.057 0.061 0.062| 0.057 0.062 0.064 0.066 0.061 0.065 0.061 0.062 0.061 0.063
```

Reconstruction error is flat at 0.057-0.066 while success swings by 0.567. Phases
1-5 win only because a short leading partial chunk reproduces the opening exactly;
the pre-registered prediction that separates "opening transient" from "periodic
alignment" — that phases 9-15 fall back to phase 0's band — held exactly.

**Exact-prefix test** (grid held at phase 0; the first N demo actions executed
verbatim), j=3, n=60:

| N | 0 | 2 | **5** | 10 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| SR | 0.367 | 0.483 | **0.900** | 0.900 | 0.883 | 0.917 | 0.867 |

Five control steps — 0.1 s at 50 Hz — is the whole effect.

**Why those five steps.** Measured on the demonstrations themselves:

- chunk 0 carries **2.2-2.5×** the interpolation error of the average chunk
- inside the first 5 steps the per-dimension error is **13-26×** higher:

| dim | what it is | first 5 steps | rest |
|---|---|---|---|
| 2 | pelvis RZ | 0.1376 | 0.0053 |
| 1 | pelvis Y | 0.1254 | 0.0077 |
| 0 | pelvis X | 0.0928 | 0.0074 |

  all three floating-base DOFs are in the top six
- the action moves **3.48×** faster there than later (mean |a[t]−a[t−1]| 0.0375 vs 0.0108)

Control is absolute joint position (`bigym/loco/env.py:163`, `action_mode: str = "absolute"`),
so the robot is simply sent to the wrong body pose while it repositions after reset.
The failure signature matches: **88%** of ZoomQ's failed episodes are timeouts —
never grasped — not plate drops.

## 4. The repair: commit less than you plan

`--mode execlen` commits the same j-knot 16-step skeleton but executes only the
first E steps before replanning:

| | E=1 | **E=2** | E=4 | E=8 | E=16 (today) |
|---|---|---|---|---|---|
| r=0 (j=2) | 0.867 | **0.833** | 0.500 | 0.117 | 0.100 |
| r=1 (j=3) | 0.867 | **0.867** | 0.533 | 0.333 | 0.367 |
| r=2 (j=5) | 0.867 | 0.900 | 0.817 | 0.867 | 0.833 |

E=1 reproducing raw replay exactly (0.867) is the implementation's self-check:
offset 0 of every plan is a knot, so E=1 must be the unprojected trajectory.

`execution_length` is now a config knob (`train_cqn_as_bigym.py`, four sites —
the eval loop has its own replan test and its own plan index, and patching only
the training pair would have trained an arm at E=2 and evaluated it at E=16
without any error). Default `E = action_sequence`, byte-identical to before.

**And the temporal ensemble does the same thing a different way.** Replaying the
same skeletons through `utils.TemporalEnsembleControl` (replan every step, ~16
near-uniformly weighted overlapping plans, gain 0.01):

| round | j | open-loop | ensemble |
|---|---|---|---|
| 0 | 2 | 0.100 | **0.617** |
| 1 | 3 | 0.367 | **0.817** |
| 2 | 5 | 0.833 | 0.900 |
| 4 | 16 | 0.867 | 0.867 |

So the 0.100 ceiling belongs to `temporal_ensemble=false`, not to the knot set.
An agent replanning every step is not committing a skeleton and cannot be stopping
early on one, which is why E=2 — which keeps the commitment — is the interesting
repair and the ensemble is only a diagnostic.

---

## 5. Phase 9: does ZoomQ learn once the protocol stops masking it?

All arms on the **stock protocol** (`temporal_ensemble=true nstep=1`), successful
training rollouts in the first 45 episodes, two seeds each:

| arm | fixes | s1 | s2 |
|---|---|---|---|
| `zqE` | neither | 1 | 0 |
| `zqES` | exec_cond_skeleton (A) only | 0 | 0 |
| `zqEB` | refine_target=percell (B) only | 0 | 2 |
| **`zqEA`** | **A + B** | **4** | **7** |
| `zqFull` | `adaptive=false`, full 16 knots | 1 | 0 |
| baseline stock | — | 23 / 21 / 16 / 29 | |
| `ctl` (commit-16) | — | 0 / 1 / 0 / 2 | |

At 90 episodes: `zqEA_s1` = 12, `zqEB_s2` = 16, baseline = 35-56.

**ZoomQ is not broken.** Every arm's first success lands at episode 20-21, with
`episode_reward` exactly 1.0. It is roughly **3-5× slower** than the baseline, not
dead — 4-7 against 16-29 at 45 episodes, 12-16 against 35-56 at 90.

**Only the combination moves it.** A alone (0, 0) and B alone (0, 2) do nothing;
together they give 4 and 7. An interaction, not a sum. n=2 seeds, so this is a
strong hint and not yet a result.

**Representation is not the binding constraint under this protocol.** `zqFull`,
forced to descend to all 16 knots, equals `zqE`, which executes 2 knots ~89% of
the time: 1 and 0 either way. Consistent with the ensemble ceilings, where r=0 is
0.617 against r=4's 0.867 — a factor of 1.4, not 50.

## 6. The stopping rule is still dead — on every arm

| arm | δ/κu p50 | δ/κu p90 | depth_share 0 / 1 / 2 / 4 |
|---|---|---|---|
| `zqEA_s1` | −0.0000 | **0.0000** | 0.888 / 0.029 / 0.028 / 0.027 |
| `zqES_s1` (Stage A) | 0.0000 | **0.0000** | 0.889 / 0.028 / 0.028 / 0.027 |
| `zqE_s1` (neither) | −0.0000 | **0.0000** | 0.887 / 0.029 / 0.029 / 0.027 |
| `zqFull_s1` | −0.0000 | 0.0000 | 0.037 / 0.038 / 0.036 / 0.853 |
| (p8, older protocol) | — | 0.0147 | 0.891 / … |

The rule needs δ/κu ≥ 1.0 to fire. It reads **0.0000**, and depths 1-4 are all
equal at 0.026-0.029 — the signature of uniform ε selection, not of a rule making
decisions. Stage A changed the exec head's *input*; the target is still one shared
`bellman` per sample, so every depth converges to the same value and Δ moves
*toward* zero rather than away from it. **The anytime headline has no support.**

## 7. The training-protocol control

`temporal_ensemble=false nstep=16` — which ZoomQ needs, because committing a chunk
is the method — costs stock CQN-AS, with no ZoomQ code involved at all:

| arm | succ @45 eps | @90 | @160 |
|---|---|---|---|
| baseline stock ×4 | 23 / 21 / 16 / 29 | 55 / 46 / 35 / 56 | 112 / 94 / 80 / 105 |
| `ctl` s1 | 0 | 0 | 0 |
| `ctl` s2 | 1 | 9 | 17 |
| `ctl` s3 | 0 | 0 | — |
| `ctl` s4 | 2 | 3 | — |

Four seeds now: one reaches 17/160, three are at or near zero. So **0/160 is a
normal outcome for this protocol**, and any ZoomQ zero measured under it — which
is every p7 and p8 arm — is uninterpretable. Those twelve and four arms were
retired for that reason.

`zqE2` (commit + `execution_length=2` + `nstep=2`) scored **0 in 69-73 episodes**,
despite a measured replay ceiling of 0.833 at r=0. The ceiling did not transfer.

---

## 8. Corrections to earlier claims in this programme

| claim | status |
|---|---|
| "Gate 1 passes on move_plate: j=3 reaches 109% of the j=16 ceiling" | **wrong** — measured under the `keep_first=False` off-by-one; at the deployment phase j=3 is 42% |
| "ZoomQ is 0/83 episodes vs baseline 11/40, p ≈ 2.5e-12, so a second defect exists" | **wrong** — read at 10-15 episodes per arm; every ZoomQ arm's first success is at episode 20-21 |
| "the deployment protocol is a null (0.68 vs 0.64, n=25)" | **stands**, but only for deployment — `nstep` is inert under `eval_only`, and the training side costs 85-100% |
| "the exec head is action-blind at round 0 (`dQ/da = 0.000000e+00`)" | **stands** |
| "gripper smearing / `hold_dims` explains the zero" | **refuted** (±5pp at n=20) |
| "ZoomQ fits demonstrations worse" | **refuted** — the `bc_margin` gap is window geometry; threshold-free `bc_fosd` is 34% *lower* |

## 9. Incident: the filesystem filled and stopped everything

`/mnt/workspace` reached 10 T of 10 T with 0 bytes available. All 16 live arms
stopped within two minutes at 12:00-12:02 with **no error in any log**. Snapshots
(~450 MB each) and `train.csv` survived, so nothing above was lost.

Two mechanisms worth recording:

- **A run dir is ~250 GB and is almost entirely replay buffer.** For
  `p2_fc_base_s1`: `buffer/` 142 GB + `demo_buffer/` 115 GB, against `snapshot.pt`
  431 MB and `train.csv` 40 KB. Launch gates that check only `free -g` — as
  `scripts/launch_phase*.sh` did — will start arms onto a full disk.
- **`scripts/watchdog.sh`'s DONE guard fails open.** It shells out to python to
  read `train.csv` with `|| echo 0` as the fallback; with the disk full that call
  fails, a 100,979-frame baseline reads back as 0 frames, and the watchdog
  relaunches it every cycle, each relaunch dying immediately.

`gates/ram_guard.sh` was **not** the cause — it only stops arms at or beyond
`DONE_FRAMES=99000`, and the dead arms were at 17K-76K.

## 10. What the evidence does and does not support

**Established.** Committing 16 steps open loop is a root cause, at three
independent levels: the ceiling (0.100 vs 0.867 for a perfect policy), the control
experiment (stock CQN-AS drops from 80-112 to 0-17 on that switch alone), and the
repair (E=2 restores 0.833; the ensemble restores 0.617).

**Established.** The stopping rule cannot fire, on any arm, at any point in
training, and Stage A does not change that.

**Not established.** Why ZoomQ is still 3-5× slower than the baseline on the stock
protocol, where the representation ceiling is 0.617 and the protocol confound is
gone. `zqEA`'s advantage over the other three cells is n=2 and needs replication
before it means anything.

---

### Reproducing the replay numbers

```bash
cd /mnt/workspace/zoomq
P=/mnt/workspace/anchorq/.venv/bin/python
export HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local MUJOCO_GL=egl

# corrected Gate 1
$P gates/gate1_replay.py --task move_plate --era floating --hold 0 --keep-first \
   --mode full --j 2,3,5,9,16 --num-demos 60 --workers 6 --phase 0 \
   --cache-root /mnt/workspace/zoomq/demos/.bigym --out gates/gate1fix_all.json

# error budget / phase sweep / exact prefix / ensemble / execution length
./scripts/phase_sweep.sh
./scripts/prefix_and_gate1.sh
./scripts/ensemble_ceiling.sh
./scripts/execlen_ceiling.sh

# training decision counters
./scripts/p10_readout.sh
```
