#!/bin/bash
# Phase 9 — retest every ZoomQ fix under the protocol that lets the BASELINE work.
#
#   ./scripts/launch_phase9.sh [--dry-run]
#
# WHAT PHASE 8 ESTABLISHED. Two independent defects, not one. First 45 episodes on
# move_plate, successful training rollouts:
#
#                        CQN-AS          ZoomQ
#   stock protocol       23 / 21 / 16    0 / 0      <- this cell had never been read
#   nstep=16, no ens.     0 /  1          0 / 0 / 0 / 0
#
#   (1) THE TRAINING PROTOCOL. nstep=16 + no temporal ensemble takes the SAME
#       CQN-AS critic on the SAME task from 16-23 successes to 0-1. An order of
#       magnitude, and it floors one seed outright. Every ZoomQ arm ever run has
#       been under it, because config_zoomq_bigym.yaml hardcodes it.
#       NB this is the TRAINING side. An earlier frozen-weight test found
#       temporal_ensemble to be a null at DEPLOYMENT (0.68 vs 0.64, n=25) -- and
#       that was correct, because `nstep` is inert under eval_only. The training
#       side was never isolated until p8_mp_ctl.
#   (2) ZoomQ ITSELF. With the protocol confound removed, ZoomQ is still 0/89
#       episodes where CQN-AS gets 16-23. The protocol does not explain ZoomQ's
#       zero; the two defects stack.
#
# WHY THIS WAVE. Every ZoomQ fix so far -- exec_cond_skeleton (Stage A),
# refine_target=percell (Stage B) -- was measured inside a protocol that flattens
# the baseline to 0-1. There is no measurable headroom in that regime, so those
# nulls are uninterpretable. Retest them where the baseline demonstrably works.
#
# THE ARMS (8), all on the STOCK protocol (temporal_ensemble=true, nstep=1):
#   zqE   ZoomQ as shipped              -- the reference for this protocol
#   zqES  + exec_cond_skeleton          -- Stage A
#   zqEB  + refine_target=percell       -- Stage B
#   zqEA  + both                        -- the full candidate
# Two seeds each. Baseline reference in this protocol: 16-23 successes / 45 eps.
#
# PRE-REGISTERED DECISION RULE (successful training rollouts in the first 45
# episodes, read straight from train.csv episode_reward):
#   any arm >= ~8    -> that fix works once the protocol stops masking it
#   zqE ~ 0 and all fixed arms ~ 0
#                    -> ZoomQ's decision structure is the defect, independent of
#                       both the protocol and the backup. That is the clean
#                       negative result, and it is the one zoomq.tex 5's Gate 2
#                       already writes the decision rule for.
#
# COST. 8 arms; measured ~56 GB each at 100K frames, 756 GB free at launch, and
# the decision needs only ~45 episodes (~10K frames, ~2.5 h), not 100K.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"
SKEL="zoomq.exec_cond_skeleton=true"
PERCELL="zoomq.refine_target=percell"

ppu=0
arm() {
  local name="$1" seed="$2"; shift 2
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${free_gb:-0}" -lt 150 ]; then
    echo "SKIP $name: only ${free_gb} GB available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT $STOCK \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 4
}

for s in 1 2; do
  arm "p9_mp_zqE_s${s}"  "$s"
  arm "p9_mp_zqES_s${s}" "$s" $SKEL
  arm "p9_mp_zqEB_s${s}" "$s" $PERCELL
  arm "p9_mp_zqEA_s${s}" "$s" $SKEL $PERCELL
done

echo "phase 9 launched: 8 arms on move_plate under the STOCK protocol"
