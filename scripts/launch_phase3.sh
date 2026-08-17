#!/bin/bash
# Phase 3 — the decisive arms for "why is ZoomQ at SR 0.00".
#
#   ./scripts/launch_phase3.sh [--dry-run]
#
# WHAT PHASE 2 LEFT AMBIGUOUS
# ---------------------------
# Offline eval at 10K frames, 25 episodes each:
#     take_cups  base 0.08 / 0.04   zqA 0.00 / 0.00
#     put_cups   base 0.12 / 0.12   zqA 0.00 / 0.00
# and every ZoomQ eval episode ran to the EXACT time limit (1050.0 / 850.0),
# i.e. zero early terminations in 100 episodes.
#
# Two things stop that from being an answer:
#
#   (a) 0/25 against a base rate of 0.08 has p = 0.92^25 = 0.12. Per arm that is
#       not significant. Only the four arms together are.
#   (b) base and zqA differ in THREE variables, not one: the ZoomQ algorithm,
#       temporal_ensemble (true -> false) and nstep (1 -> 16). The stock protocol
#       was chosen for the baseline on purpose (CQN-AS's own ablation reports
#       temporal ensembling helps), but it means the 0.00 cannot be attributed.
#
# THE MEASURED MECHANISM THESE ARMS TEST
# --------------------------------------
# depth_share_0 = 0.897 on every ZoomQ arm: the stopping rule never fires (the
# exec-head tie, exec_q spread 5e-4 across depths), so ~90% of training actions
# and ~100% of EVAL actions are the ROUND-0 skeleton — two knots at t=0 and
# t=15, linear fill in between.
#
# On these tasks the gripper dims are strictly binary in the demonstrations:
#     take_cups dims 14,15   intermediate fraction 0.0000
#     put_cups  dims 14,15   intermediate fraction 0.0021
#     move_plate dim 13      intermediate fraction 0.0394
# and on take_cups the transition is a PERFECT STEP: 0 intermediate steps over
# 33 demos x 4 transitions each. A 2-knot linear fill cannot represent a step.
# Measured over the real demo chunks, under round-0 linear fill:
#     7.2-8.9% of chunks contain a gripper state change, and in those chunks
#     75% of timesteps command a physically meaningless half-closed gripper.
#
# hold_dims applies ZERO-ORDER HOLD to the listed dims instead (bigym_src/
# zoomq.py::_fill returns `lo` there). Measured effect on the same chunks:
#     take_cups  gripper intermediate 0.0547 -> 0.0000
#     put_cups   gripper intermediate 0.0678 -> 0.0018
#     move_plate gripper intermediate 0.1369 -> 0.0290
# Note chunk RMSE gets WORSE (0.0372 -> 0.0533 on take_cups): L2 prefers the
# ramp. That is the point — exec_match_tol, skeleton_rmse and Gate 0's j* are
# all L2 criteria and all of them rate the unusable fill as the better one.
#
# THE ARMS
# --------
#   ctl — stock CQN-AS with temporal_ensemble=false nstep=16. Isolates (b).
#         Everything else identical to the Phase 2 `base` arm.
#   zqH — zqA plus hold_dims only. A ONE-VARIABLE delta from an arm already
#         measured at 0.00, so a recovery is attributable.
#   zqA — ZoomQ as the paper describes it. On move_plate this does not exist
#         yet; move_plate is the ONLY task whose baseline is reproduced
#         (0.68 at 40K vs published 64.0 +- 7.5), so it is the only place a
#         ZoomQ number can be read against a trusted reference.
#
# PRE-REGISTERED DECISION RULE (25-episode offline eval, >= 30K frames)
# --------------------------------------------------------------------
#   ctl ~ base                  -> the execution protocol is exonerated; the
#                                  0.00 belongs to ZoomQ.
#   ctl ~ 0.00                  -> nstep=16 / no temporal ensembling is the
#                                  cause and ZoomQ is exonerated. The whole
#                                  Phase 2 comparison would then be void.
#   zqH >> zqA on tc/pc         -> round-0 gripper smearing confirmed as the
#                                  dominant cause.
#   zqH ~ zqA ~ 0.00            -> the gripper is a contributor, not the cause;
#                                  next suspect is the round-0 skeleton itself
#                                  (maxabs_p90 = 0.93 on the continuous dims).
#   move_plate zqA ~ 0.00       -> ZoomQ fails wherever a grasp is required,
#     while base = 0.68            independent of task difficulty.
#
# COST — 11 arms x ~2.5 cores = ~28 cores on top of the ~113 already running,
# so ~141 of 184. CLAUDE.md 2.3 puts the sim-work ceiling at ~150.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
# The matched-execution protocol the ZoomQ configs already use.
MATCHED="temporal_ensemble=false nstep=16"

# Spread over the cards the eval daemons are NOT on (they hold 10-15).
ppu=0
next_ppu() { ppu=$(( (ppu + 1) % 10 )); }

# arm <name> <config|-> <task> <seed> <extra...>
arm() {
  local name="$1" cfg="$2" task="$3" seed="$4"; shift 4
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=10000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task="$task" $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  next_ppu
  sleep 2
}

# ------------------------------------------------------- move_plate (dim 13) --
# Baseline reproduced: 0.64/0.88/0.76/0.84 (s1), mean 0.68 at 40K. Published 64.0.
arm p3_mp_ctl_s1 -                  move_plate 1 $MATCHED
for s in 1 2; do
  arm "p3_mp_zqA_s${s}" config_zoomq_bigym move_plate "$s"
  arm "p3_mp_zqH_s${s}" config_zoomq_bigym move_plate "$s" 'zoomq.hold_dims=[13]'
done

# --------------------------------------------------- take_cups (dims 14, 15) --
arm p3_tc_ctl_s1 -                  take_cups 1 $MATCHED
for s in 1 2; do
  arm "p3_tc_zqH_s${s}" config_zoomq_bigym take_cups "$s" 'zoomq.hold_dims=[14,15]'
done

# ---------------------------------------------------- put_cups (dims 14, 15) --
arm p3_pc_ctl_s1 -                  put_cups 1 $MATCHED
for s in 1 2; do
  arm "p3_pc_zqH_s${s}" config_zoomq_bigym put_cups "$s" 'zoomq.hold_dims=[14,15]'
done

echo "phase 3 launched: 11 arms"
