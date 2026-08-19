# ZoomQ — brief for a fresh session

You are picking up a research programme that has run for several days and produced a
large amount of evidence, most of it negative. Your job is to think about it
**independently**: the person who assembled this brief was wrong about the mechanism
seven times and wrong about an experimental design once, so treat every framing below
as evidence to re-examine rather than as conclusions to build on. Where a claim is
marked MEASURED, the number is real and reproducible; where it is marked INFERRED, it
is someone's story about the numbers and is fair game.

---

## 1. The setting

**CQN-AS** is the baseline: a critic-only coarse-to-fine (C2F) RL agent for action
*chunks*. For each of T=16 timesteps x D=15 action dims it runs L=3 amplitude levels
x B=5 bins, zooming in on the chosen bin at each level (125 effective bins/dim). The
critic is C51 (51 atoms on [-2,2], atom spacing dz=0.08), dueling. Training is online
RL from demonstrations: one loop, no offline phase, `action_repeat=1` and
`update_every_steps=1` so 1 env frame = 1 gradient step, with a 256-sample online
batch plus a 256-sample demo batch carrying a BC loss (`bc_lambda=1.0`, FOSD term +
hinge margin term with `bc_margin=0.1`) every step.

**ZoomQ** (`~/projects/anchor_q/zoomq.tex`, section 3) is the fork under test. It adds
a *temporal* coarse-to-fine axis on top: the K=16 chunk is committed over R=5 dyadic
rounds with knot sets `[[0,15],[7],[3,11],[1,5,9,13],[2,4,6,8,10,12,14]]`
(n_r = 2,3,5,9,16), each round refining inside a residual window around the
interpolation of the previously committed knots (`w_schedule = [1.0, 0.15, 0.08,
0.04, 0.02]`). Its claimed contribution is **anytime execution**: every round is
already an executable skeleton, and a learned stopping rule ends refinement early when
further refinement would not add value. The rule is `refine iff delta_r > kappa*u_r`,
where delta_r is the gain in an execution-value head Q^exec and u_r is an uncertainty
scale.

**Task:** BiGym `move_plate`, floating-base H1 (NOT the HOMIE stack — demos replay
0/60 under HOMIE vs 0.867 floating). 300-step episode cap; a failed episode runs the
full 300 steps; `_fail()` fires only when a plate touches the floor.

**Code:** `/mnt/workspace/zoomq` on the host `ali` (`ssh ali`). Algorithm in
`third_party/CQN-AS-G1/bigym_src/{zoomq.py, cqn_as.py}` on branch `zoomq`; trainer
`train_cqn_as_bigym.py`; configs in `cfgs/`. Papers at
`~/projects/anchor_q/{main.tex, zoomq.tex}`. Everything is pushed to
`github.com:zzchhh9/zoomq.git`.

---

## 2. The bottom line, as measured

Successes in the first 90 training episodes, **1x-demo-buffer arms only** (see trap 1
below for why that qualifier matters), per seed:

| arm | config | per-seed | rate |
|---|---|---|---|
| CQN-AS baseline | — | 55, 46, 35, 56 | **0.533** |
| zqEAT | A+B+tie-break | 11, 8 | 0.106 |
| zqEAd | A+B+full depth | 15, 0 | 0.083 |
| zqEAw | A+B+wide windows | 5, 4 | 0.050 |
| zqTR / zqTM | tie-break random / middle | 0,0 / 0,0 | 0.000 |

where **A** = `zoomq.exec_cond_skeleton=true` (the exec head sees the skeleton) and
**B** = `zoomq.refine_target=percell` (per-cell TD targets instead of a broadcast).
Only A+B together ever moved anything; A alone and B alone score 0-2.

**A 5x gap, and six interventions have not closed it.**

---

## 3. Established, with the measurement

1. **The stopping rule cannot fire, by construction.** `exec_ce` regresses *every*
   depth onto ONE shared `bellman` label, so Q^exec is identical at every depth and
   delta is identically 0. Measured `delta_over_kappa_u` p90 = **1.3e-05** against the
   1.0 it needs, on every arm at every checkpoint. Separately, `u_r` uses the C51
   spread whose floor is one atom (0.084) against an across-depth delta of <=0.015 —
   so the rule is arithmetically dead even if delta were fixed. `depth_share_0` = 0.89
   and depths 1-4 sit exactly on the annealed eps floor. **MEASURED.**

2. **A discretisation level learns only when it can pick the right bin more than half
   the time.** Sweeping the hinge margin m in {0.02, 0.05, 0.1, 0.3} x window width,
   the flat-to-open boundary sits at held-out bin accuracy **0.490-0.513** and does
   **not** move with m (boundary ratio varies 1.39x across a 15x margin range); the
   *magnitude* of the opened spread scales with m. This follows in two lines from the
   hinge: `E[loss](s) ~ p*clamp(m-s,0) + (1-p)(m+s)`, `d/ds at 0+ = 1-2p`, so flat is
   optimal iff p <= 1/2. **MEASURED, and it is a known principle, not a new law.**

3. **ZoomQ's residual-window schedule puts its finest level below that boundary by
   construction.** L2 bin width is `2w/125`, so per round it is
   0.016 / 0.0024 / 0.00128 / 0.00064 / 0.00032 against the residual each round must
   remove (`skeleton_rmse_r` in train.csv: 0.0652 / 0.0361 / 0.0225 / 0.0138). Ratios
   at rounds 1-4 are 0.023-0.037; CQN-AS is 0.246 at every cell. Consequence: L2
   spread 0.0018 atoms (zqEA) vs CQN-AS 4.42, and 77-85% of L2 cells sit under the
   1e-4 tie threshold vs CQN-AS's 0.007%. **MEASURED.** The narrower the window, the
   finer the bin, the lower the predictability, the flatter the level — the method's
   own refinement mechanism disables its own discrimination. **INFERRED framing.**

4. **Round 0 is flat too, on a lattice identical to CQN-AS's.** Round 0 has no parent
   so its window must span [-1,1]; its L2 bin is 0.016 and its bin/residual is 0.246,
   the same as CQN-AS — and its L2 spread is still 0.0027 with bin accuracy 0.494
   against CQN-AS's 4.42 / 0.916. No `w_schedule` value fixes it (w = 0.2, 0.5, 1.0
   all leave it at 0.001-0.006). On a *fresh* critic, training the L2 loss on round-0
   cells only opens it (spread 3.49, acc 0.968) while training all rounds does not
   (0.00042, 0.338) — so the deep-round L2 loss terms suppress the shared head's
   amplitude. **MEASURED.** Note the r0-only variant overfits: held-out acc is 0.54.

5. **The exec head is 46% of the critic loss and receives no reward.**
   `critic_loss = critic_lambda*refine_lambda*refine_loss` then
   `+ exec_lambda*exec_loss` — `exec_lambda=0.1` is NOT multiplied by
   `critic_lambda=0.1`, so it enters at the same coefficient as TD and measures 46%
   of the total. Meanwhile `exec_valid` zeroes any sample whose 16-step chunk is
   padded, and **every** reward-bearing transition is padded: 269/269 and 267/267.
   Under `refine_target=chain` the exec head is also the sole bootstrap carrier, so
   reward cannot enter the backup at all — which is why the chain arms score 0-1 and
   the percell arms 2-7. **MEASURED.**

6. **Executing the round-0 skeleton costs 1.6x in action error**, and the temporal
   ensemble does not rescue it: same weights, blended post-ensemble MAE is 0.0153
   (full depth) vs 0.0288 (round-0 skeleton). Raw chunk MAE at matched training:
   CQN-AS 0.0063, zqEA 0.0210, zqE 0.0713. ZoomQ's oracled lattice floor is 0.00277
   — **finer** than CQN-AS's 0.00453 — so the lattice can represent better and the
   critic realises less of it (CQN-AS lands at 1.7x its floor, zqEA at 4.5x, zqE at
   15.9x). **MEASURED.**

7. **The opening transient dominates this task.** Executing just the first 5 demo
   actions verbatim takes a 3-knot replay from 0.367 to 0.900; chunk 0 carries
   2.2-2.5x the average interpolation error and 13-26x on the three floating-base
   DOFs, where the action moves 3.48x faster than later. 88% of ZoomQ's failures are
   timeouts (never grasped); across 400 rollouts terminal reasons were
   {timeout 301, success 97, plate_floor 2}. **MEASURED.**

8. **Corrected Gate 1** (n=60 demo replay, deployment phase, no network):
   round 0 (j=2) 0.100, round 1 (j=3) 0.367, round 2 (j=5) 0.833, round 3 (j=9)
   0.900, round 4 (j=16) 0.867 — against an unprojected 0.867. Under the temporal
   ensemble round 0 rises to 0.617. The published table (j=2 0.467, j=3 0.817) was
   measured under an off-by-one that shifted the chunk grid. **MEASURED.**

---

## 4. Refuted — do NOT re-propose these

| hypothesis | the number that kills it |
|---|---|
| ZoomQ's critic is globally flat | pooled spread 4.85-5.77 atoms vs baseline 5.63-8.33 — comparable. Only the finest level is dead |
| The exec head drags the shared trunk toward action-invariant features | gradients are orthogonal (cos(refine,exec)=0.0032, cos(exec,bc)=-0.0563) and ZoomQ's TD gradient into the trunk equals the baseline's (0.083 vs 0.081) |
| The tie-break (bare argmax picks bin 0 where the baseline randomises) is the cost | randomising makes frozen-weight MAE *worse* (0.07127 to 0.07595, 8/8 seeds, 9.5 sigma) and both `random` and `middle` score 0/180 in training |
| Naive bin-width geometry explains the flat level | bins span 50x across rounds while L2 spread spans 3x, and r0 is about equal to r4 |
| The head is round-blind | `cond = [mids, p, w, onehot_t, onehot_r]` (zoomq.py:660) — round enters three times; perturbing only the round signal moves L2's Q by 25.13 atoms |
| The BC target churns | L2 target-bin change is exactly 0 over 22 checkpoints x 61,440 pairs; training teacher-forces `parent_prediction` on the demo action (zoomq.py:735). Round 0's L2 targets are bit-identical to CQN-AS's on 7680/7680 |
| BC never reaches the finest level | its per-sample head gradient there is 2.77e-4 vs 4.93e-5 at L0 — 5.6x **larger**; no level mask; dueling mean-subtraction leaves the spread bit-identical |
| Action accuracy orders the arms | p13 executes a strictly better action (MAE 0.0250 vs 0.0329, p95 0.0636 vs 0.1018) than zqEA and scores lower |
| Trajectory smoothness orders the arms | the BEST arm is the ROUGHEST (CQN-AS jerk2 = 1.80x the demos); the exactly-linear round-0 skeleton beats full depth |
| The temporal ensemble destroys the full-depth chunk | blend/raw = 0.780 for full depth vs 1.469 for the skeleton — the blend *helps* full depth |
| Knot precision is the binding constraint | at ZoomQ's measured knot MAE (0.049) an open-loop j=16 chunk still scores 0.600, and under the ensemble the effect is inside the seed spread |

---

## 5. Methodological traps this programme fell into — check for them before trusting any number

1. **Crash restarts inflate the demo buffer.** Each restart re-appends the whole demo
   set. `p9_mp_zqEA` — the control for phases 11, 12 and 13 — had 15 restarts, 269
   demo files and 42,337 buffer entries against p13's 3 / 71 / 9,988: **4.3x the
   training data.** Always run `ls runs/<arm>/demo_buffer | wc -l` before using an arm
   as a control.
2. **Never score before about 90 episodes.** First successes across this family land
   between episode 11 and 54. A 45-episode gate once closed two episodes before an
   arm's take-off and produced a confident "11x regression" that was a null.
3. **Variance lives at the seed, not the episode.** `zqEB` scores 0/85 and 18/93 on
   one identical config and one identical buffer. Pooling episodes into a Poisson
   test gave P = 2e-04 where a seed permutation gave P = 0.33-0.60. With n=2 seeds
   the permutation floor is P = 0.33, so nothing is readable.
4. **`eps_depth` anneals 0.3 to 0.05 over 20,000 steps**, so `depth_share_*` and clamp
   rates move on their own. At 1200 frames two different arms reported
   `depth_share_0` bit-identical at 0.7384001855055491. **Only compare at matched
   frames.**
5. **Frozen-weight sweeps are hypothesis-generating only.** `tie_break=middle` had
   the best frozen-weight MAE of three options and scored 0/180 in training.
6. Config names lie: `exec_lambda: 0.1` is 46% of the loss; "FLOPs are unchanged"
   for `adaptive=false` is false (`early_exit` at zoomq.py:966 means adaptive=true
   computes 2 of 5 rounds, adaptive=false computes 5 — 2.5x the descent).
7. Operational: reading a live `snapshot.pt` gives torn-file errors (use
   `eval_snapshots` or copy first); importing `cqn_utils`/`bigym_src.zoomq` at module
   scope before the first encoder forward **segfaults** (import lazily,
   `OMP_NUM_THREADS=8`); parse `train.csv` by header name; `/mnt/workspace` filled to
   10T/10T once and killed 16 arms with no traceback in any log.

---

## 6. Open questions, ranked by how much the answer would change the picture

1. **Is there a single monotone action-error to success curve that fits the baseline
   AND the ZoomQ family?** CQN-AS is at chunk MAE 0.0063 giving 0.533 and ZoomQ at
   0.021-0.071 giving 0.05-0.11. The 4x error gap and the 5x success gap may be *one*
   relation, with every within-ZoomQ comparison simply underpowered. Nobody has
   fitted this. If it holds, there is no mystery to explain and the whole question
   becomes "why is ZoomQ's critic 4x less accurate on a *finer* lattice".
2. **ZoomQ has only ever been meaningfully tested on ONE task.** CQN-AS itself scores
   ~0 on take_cups (0/96, 1/96), put_cups (2/119, 0/118), flip_cup (0/91, 0/82) and
   dishwasher, so the 12 ZoomQ arms run there say nothing. The only *other* task where
   the baseline demonstrably works is **saucepan_to_hob** (65/109, 90/112, 58/99) —
   and **ZoomQ has never been run on it.** move_plate's specific failure mode (the
   opening transient; 88% timeouts, never grasped) is maximally hostile to coarse
   temporal skeletons, so every conclusion here may be a property of the task.
3. **Does the dyadic temporal axis buy anything at all?** ZoomQ's coarsest amplitude
   level is *more* opinionated than the baseline's (16.96 vs 8.42 atoms) — the coarse
   temporal structure is being learned better — while the fine structure is worse. Net
   worse. Is there a regime (larger K, smoother tasks, a tight compute budget) where
   the trade flips? K=16 gives 5 rounds against 3 levels, which is where the dyadic
   argument is *weakest*; nobody has tried K=32 or 64.
4. **Never tried:** fixing `exec_valid` so reward-bearing transitions reach the exec
   head; making `exec_lambda` actually 0.1 by multiplying it by `critic_lambda`;
   depth-specific exec targets (the change that would make delta non-zero — the
   paper's own section 3.3 names it and also predicts the deep heads then starve,
   since they see about 2.7% of data each, all from eps); `agent.levels=2` (measured
   on a fresh critic to open round-0's finest level to 3.00 atoms / 0.802 accuracy,
   never launched as an arm); per-cell resolution driven by each cell's own residual;
   annealing the number of levels as the residual shrinks.
5. **Do NOT do:** `exec_lambda=0` — it is the exec head's only gradient
   (zoomq.py:1459), so zeroing it silently turns A into a no-op and the arm becomes
   percell-only. And do not stack an L2 mask with forced full depth: the mask works by
   *not training* rounds 1-4's L2 while full depth *executes* exactly those knots.

---

## 7. What you can use

- `gates/gate1_replay.py` — demo replay with `--mode {raw,full,zoh,chunk,perturb,ensemble,execlen}`,
  `--phase`, `--exact-prefix`, `--knot-noise`, `--exec-len`. Faithful reimplementation
  of `TemporalEnsembleControl` included and verified.
- `gates/fast/exec_mae.py` — executed-action error under either depth policy, same weights.
- `gates/reanalyse_matched_buffers.py` — the scoreboard with buffer-tier grouping.
- `/root/gap/{critic_discrim,critic_localize,oracle,traj}.py` — per-level/per-round
  spread, bin accuracy, tie fraction, and an oracle decomposition of the action error.
  `oracle.py` re-implements `descend`/`get_action` bit-identically to the shipped path.
- `/root/law/{mask4.py, analyse.py, go_law.sh}` — short-optimisation harness on a fresh
  critic with `--geom`, `--wdeep`, `--margin`, `--l2mask`, `--levels`, held-out split.
  This is the tool that measured the p=1/2 boundary.
- `reports/REPORT_2026-08-18_gate1_and_phase9.md` and
  `reports/GROK_2026-08-19_shared_head_skeptic.md`.
- Roughly 40 trained arms under `runs/` with `train.csv`, `snapshot.pt` and (for
  p9/p13) `eval_snapshots/`.
- Throughput about 1 frame/s per arm, ~300 frames/episode; 184 cores, 1.8 TB RAM,
  16 PPUs (NOT NVIDIA — source `/usr/local/PPU_SDK/envsetup.sh` first).

---

## 8. What would count as progress, and what would not

**Would not:** another single-flag arm on move_plate compared against a 2-seed control.
The programme has run six of those; four were nulls, two were negative, and the
statistics could not have detected a real effect anyway.

**Would:** anything that either (a) makes the anytime claim testable on its own terms
by giving delta a chance to be non-zero, (b) establishes whether the whole picture is
one task's pathology by running the arms that matter on saucepan_to_hob, or (c) fits
the one-curve hypothesis in 6.1 and thereby removes several "mysteries" at once.

**Also legitimate:** concluding that the honest deliverable is the negative result, and
saying precisely what it is a negative result *about* — the anytime headline as
specified, the dyadic decomposition in general, or this implementation. The evidence
for (i) is strong and structural; for (ii) it is confounded by one task; for (iii) it
is a list of six specific defects, four of which are still unfixed.

---

## 9. Your task

Think about this from scratch. You may conclude that the framing in section 3 is
wrong, that the ranking in section 6 is wrong, or that the whole programme is asking
the wrong question. What you should *not* do is re-run anything in section 4, or
propose an experiment whose statistics could not detect the effect it is looking for.

Two specific invitations:
- The most interesting unexplained fact is 6.1 read against 3.6: ZoomQ has a
  **finer** lattice floor than CQN-AS (0.00277 vs 0.00453) and realises **less** of it
  (4.5x its floor vs 1.7x). Why would adding a temporal refinement axis make a critic
  worse at using its own action space?
- Everything here is about a *critic-only* method discretising an action chunk. Ask
  whether the mechanism in 3.2-3.3 says anything about discretised action
  representations in general — action tokenisers, VQ codebooks, distributional atom
  counts — where the same "label finer than the model's predictive resolution" problem
  would arise.
