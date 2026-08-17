#!/bin/bash
# Phase 8 — localise the BEHAVIOURAL failure: backup, structure, or protocol?
#
#   ./scripts/launch_phase8.sh [--dry-run]
#
# THE FINDING THIS WAVE DECIDES. At matched frames (<=9.5K) on move_plate the
# CQN-AS baseline logs 47/114 successful training rollouts across 3 seeds; ZoomQ
# logs 0/380 across 12 arms. The knot-selecting argmax runs on the PER-CELL
# heads, whose TD target is the chain broadcast (<=4 distinct distributions per
# sample over 768 cells, measured to collapse onto `bellman` +0.0005 Q) — so TD
# never tells the policy which action is better and selection is BC-only.
# Arm C triangulates the same spot: healthy per-bin Q, near-demo selection,
# full-depth execution — still 0.00. The one thing every ZoomQ variant shares is
# this backup. But a third suspect was never cleared: every ZoomQ arm trains
# under nstep=16 + no temporal ensemble, the baseline under nstep=1 + ensemble,
# and the one arm that isolated that (p3_mp_ctl) was culled before its first
# eval was ever read.
#
# THE ARMS (6), one variable each:
#   zqB   zqA + zoomq.refine_target=percell           (the backup, alone)
#   zqBS  zqB + zoomq.exec_cond_skeleton=true         (the full candidate)
#   ctl   stock CQN-AS + temporal_ensemble=false nstep=16
#         (the training-protocol cell of the 2x2, never yet read)
#
# PRE-REGISTERED DECISION RULE — successful training rollouts at matched frames
# by 10K (baseline reference: 47/114 ~ 41%):
#   any zqB seed > 0            -> the backup is the behavioural killer
#                                  (implementation layer)
#   zqB = 0 and ctl ~ baseline  -> the dyadic decision structure itself is
#                                  implicated (method layer)
#   ctl ~ 0                     -> the nstep-16 training protocol is the killer
#                                  and BOTH other readings are confounded
# The mechanism arms (p7 zqS/zqA) keep running untouched; their trend feed
# continues via gates/mechanism_trend.sh.
#
# EVAL_EVERY=2000: the first snapshot is the first measurement, and train.csv's
# episode_reward gives the decision-rule counter continuously without waiting
# for offline evals at all.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
PERCELL="zoomq.refine_target=percell"
SKEL="zoomq.exec_cond_skeleton=true"
MATCHED="temporal_ensemble=false nstep=16"

ppu=10
arm() {
  local name="$1" cfg="$2" seed="$3"; shift 3
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${free_gb:-0}" -lt 200 ]; then
    echo "SKIP $name: only ${free_gb} GB available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task=move_plate $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 4
}

for s in 1 2; do
  arm "p8_mp_zqB_s${s}"  config_zoomq_bigym "$s" $PERCELL
  arm "p8_mp_zqBS_s${s}" config_zoomq_bigym "$s" $PERCELL $SKEL
  arm "p8_mp_ctl_s${s}"  -                  "$s" $MATCHED
done

echo "phase 8 launched: 6 arms on move_plate"
