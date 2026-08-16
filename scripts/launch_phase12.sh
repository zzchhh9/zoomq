#!/bin/bash
# Phase 1 (reproduce the published CQN-AS baseline) + Phase 2 (the comparison,
# on the paper's primary band). Run on ali from /mnt/workspace/zoomq.
#
#   ./scripts/launch_phase12.sh [--dry-run]
#
# EVERY CHOICE HERE IS PINNED TO anchor_q/main.tex Table 1 AND CQN-AS's
# published protocol. Read this block before changing anything.
#
# Substrate — BiGym 1, floating base. `lowerbody_policy.enabled=false` plus
#   legacy_delta base actions and no leg support. Gate 1 measured the stock
#   demonstrations replaying 0/60 under the HOMIE stack and 0.750 here.
#
# success_hold_seconds=0 — the hold criterion postdates the paper (added to this
#   fork 2026-07-16). Gate 1 measured hold=1.0 as structurally unsatisfiable on
#   these demos: it needs 50 consecutive rewarding steps and the demos carry a
#   maximum of 26.
#
# Protocol for the BASELINE arms is stock CQN-AS: temporal_ensemble=true,
#   nstep=1, use_compile=true. NOT the matched-execution control
#   (temporal_ensemble=false nstep=16) — that is a ZoomQ comparison device, and
#   CQN-AS's own ablation reports temporal ensembling HELPS on most tasks, so
#   using it as "the baseline" handicaps the reference.
#
# episode_length — LEFT AT THE TASK YAML. main.tex Table 1's "length" column is
#   NOT the episode budget: measured demo lengths at dsr=10 are Take Cups 881,
#   Put Cups 756, Dishwasher Unload Cutlery 1382, Flip Cutlery 657, Saucepan To
#   Hob 1047 outer steps, all LONGER than that column (420/425/620/500/440) and
#   all comfortably under the repo yaml (1050/850/1550/1250/1100). Overriding to
#   the paper's numbers would truncate every demonstration.
#
# in_train_eval=false + save_eval_snapshot=true — the repo's own LAUNCH RULE
#   (CLAUDE.md:362). In-train evaluation runs on the training process, and on
#   these tasks it is ruinous: 25 episodes x 1050-1550 steps is 26k-39k env
#   steps per eval against a 10k-step interval, i.e. 2.6-3.9x the training cost.
#   Snapshots are still written; evaluate them with scripts/eval_daemon.py on
#   the idle cores, at --episodes 100 (25-episode numbers carry +-10pt noise).
#
# WORKERS=8 — py-spy over 2296 samples of a live arm: 49% MuJoCo step + camera
#   render, 22% the update, and 16% in the replay DataLoader (pin_memory 10.4%,
#   do_one_step 5.7%) at the default 2 workers. CLAUDE.md §2.5 says be generous
#   with workers on a 184-core box. Keep it identical across arms — it changes
#   the sampling interleaving, hence the RNG stream.
#
# DEMO_HOME — the public release carries 40 tasks; ali's shared cache had only
#   11 extracted, and under a 3-floating-DOF directory name while the code
#   requests 4. An isolated cache avoids touching shared state.
set -euo pipefail

DRY=""
STAGE="all"
for a in "$@"; do
  case "$a" in
    --dry-run) DRY="echo [dry-run]" ;;
    --stage=*) STAGE="${a#--stage=}" ;;
  esac
done

# Demonstrations ship state-only; the pixel observations are RENDERED on first
# use, ~7.6 min per demo on the long tasks (Take Cups demos are 881 outer steps).
# Six arms per task would render the same cache six times over AND race on the
# same safetensors paths, so stage `cache` runs exactly one arm per task and
# stage `rest` starts the others once those caches exist.
#   --stage=cache   one arm per task (6)
#   --stage=rest    the remaining 24
#   --stage=all     everything (only safe once the caches are built)
in_stage() {
  case "$STAGE" in
    all)   return 0 ;;
    cache) case "$1" in p1_mp_s1|p1_sh_s1|p2_tc_base_s1|p2_pc_base_s1|p2_du_base_s1|p2_fc_base_s1) return 0 ;; *) return 1 ;; esac ;;
    rest)  case "$1" in p1_mp_s1|p1_sh_s1|p2_tc_base_s1|p2_pc_base_s1|p2_du_base_s1|p2_fc_base_s1) return 1 ;; *) return 0 ;; esac ;;
  esac
  return 0
}

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
ZQ_FIX="zoomq.exec_match_mode=online_minimal zoomq.exec_loss_norm=per_depth"

ppu=0
next_ppu() { ppu=$(( (ppu + 1) % 16 )); }

# arm <name> <config|-> <task> <seed> <eval_every> <extra...>
arm() {
  local name="$1" cfg="$2" task="$3" seed="$4" ev="$5"; shift 5
  in_stage "$name" || return 0
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY="$ev" EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task="$task" $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  next_ppu
  sleep 2
}

# ---------------------------------------------------------------- Phase 1 ----
# Reproduce the published number. This is the GATE: if Move Plate does not reach
# roughly 50% by 100K frames, the environment is still wrong and Phase 2's
# numbers mean nothing, however they come out.
#   Move Plate       CQN-AS 64.0 +- 7.5
#   Saucepan To Hob  CQN-AS 80.5 +- 13.3
for s in 1 2 3; do
  arm "p1_mp_s${s}"  - move_plate      "$s" 10000
  arm "p1_sh_s${s}"  - saucepan_to_hob "$s" 10000
done

# ---------------------------------------------------------------- Phase 2 ----
# The primary band: the four tasks where CQN-AS scores low, so there is headroom
# to measure. Published CQN-AS: Flip Cutlery 4.5, Take Cups 14.0,
# Dishwasher Unload Cutlery 19.0, Put Cups 22.5. Every other baseline (CQN,
# DrQ-v2+) sits at 0.0 on three of the four.
#
# Three methods per task:
#   base — stock CQN-AS (the reference)
#   zqA  — ZoomQ as the paper describes it (default `any` attribution). Note the
#          stopping rule provably cannot fire under `any`, so this arm is
#          "ZoomQ with the depth decision disabled".
#   zqF  — ZoomQ with the execution-head attribution fix.
for s in 1 2; do
  for spec in "tc:take_cups" "pc:put_cups" "du:dishwasher_unload_cutlery" "fc:flip_cutlery"; do
    short="${spec%%:*}"; task="${spec#*:}"
    arm "p2_${short}_base_s${s}" -                    "$task" "$s" 10000
    arm "p2_${short}_zqA_s${s}"  config_zoomq_bigym   "$task" "$s" 10000
    arm "p2_${short}_zqF_s${s}"  config_zoomq_bigym   "$task" "$s" 10000 $ZQ_FIX
  done
done

echo "stage=${STAGE} launched"
