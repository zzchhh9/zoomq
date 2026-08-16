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
# episode_length — the repo's task yamls carry BiGym-2-era budgets that are
#   2.0-2.5x the paper's for five of these tasks (they were raised for native
#   HOMIE demos collected at episode_length=15000). Pinned back to the paper's
#   lengths x demo_down_sample_rate=10. Move Plate and Reach Target Single
#   already match and are left alone.
#
# DEMO_HOME — the public release carries 40 tasks; ali's shared cache had only
#   11 extracted, and under a 3-floating-DOF directory name while the code
#   requests 4. An isolated cache avoids touching shared state.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
ZQ_FIX="zoomq.exec_match_mode=online_minimal zoomq.exec_loss_norm=per_depth"

ppu=0
next_ppu() { ppu=$(( (ppu + 1) % 16 )); }

# arm <name> <config|-> <task> <episode_length> <seed> <eval_every> <extra...>
arm() {
  local name="$1" cfg="$2" task="$3" eplen="$4" seed="$5" ev="$6"; shift 6
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY="$ev" EVAL_EPS=25 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task="$task" episode_length="$eplen" $FLOAT "$@" \
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
  arm "p1_mp_s${s}"  - move_plate      3000 "$s" 5000
  arm "p1_sh_s${s}"  - saucepan_to_hob 4400 "$s" 10000
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
  for spec in "tc:take_cups:4200" "pc:put_cups:4250" "du:dishwasher_unload_cutlery:6200" "fc:flip_cutlery:5000"; do
    short="${spec%%:*}"; rest="${spec#*:}"; task="${rest%%:*}"; eplen="${rest##*:}"
    arm "p2_${short}_base_s${s}" -                    "$task" "$eplen" "$s" 10000
    arm "p2_${short}_zqA_s${s}"  config_zoomq_bigym   "$task" "$eplen" "$s" 10000
    arm "p2_${short}_zqF_s${s}"  config_zoomq_bigym   "$task" "$eplen" "$s" 10000 $ZQ_FIX
  done
done

echo "launched: 6 Phase-1 arms + 24 Phase-2 arms"
