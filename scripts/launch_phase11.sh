#!/bin/bash
# Phase 11 — is the tie-break the cost, and is it the BIAS or the VARIANCE?
#
#   ./scripts/launch_phase11.sh [--dry-run]
#
# WHY THIS RUNS BEFORE THE READ-ONLY COUNTERFACTUAL FINISHES. The counterfactual
# (swap the rule on already-trained weights, measure chunk MAE) cannot test the
# thing that most likely matters. Both constructors zero value_head and adv_head,
# and num_expl_steps is 0, so at step 0 EVERY cell is tied -- and there the shipped
# rule sends all of them to bin 0. Measured: CQN-AS emits 64 distinct actions over
# 64 states (across-state std 0.5740), ZoomQ emits ONE (std 0.00000, every dim
# -0.992). A degenerate opening policy can only be tested by training out of it.
#
# WHAT IS MEASURED, on 64 demo states at ~10K frames:
#   fraction of per-cell decisions with across-bin Q range < 1e-4 (the tie threshold)
#     CQN-AS 0.0022%  |  zqEA 32.03% (L2 77.43%)  |  zqE 38.22% (L2 85.45%)
#   cqn_as.py:479 randomises those; zoomq.py did not -- a bare argmax returns bin 0,
#   the most negative bin, so a flat critic became a systematic negative bias.
#   Chunk MAE vs the demo action: CQN-AS 0.0078, zqEA 0.0215, zqE 0.0713.
#
# THE ARMS (6), stock protocol (temporal_ensemble=true nstep=1), two seeds each:
#   zqTR   zqE + zoomq.tie_break=random   -- the direct test. zqE scores 1,0 today.
#   zqTM   zqE + zoomq.tie_break=middle   -- separates the two things `random`
#          conflates. If middle ~ random the damage was the BIAS of picking an
#          extreme bin; if middle >> random the damage was the VARIANCE random
#          adds. Nothing measured so far distinguishes them, and the distinction
#          decides whether the fix is "randomise" or "centre".
#   zqEAT  zqEA + tie_break=random        -- does it stack on the arm that already
#          works (4,7)? zqEA's critic is the healthiest of the ZoomQ arms, so if
#          the tie-break matters at all it should still show here.
#
# PRE-REGISTERED DECISION RULE, successes in the first 45 episodes from train.csv
# episode_reward (baseline reference 23/21/16; zqE 1,0; zqEA 4,7):
#   zqTR >= 4 on either seed   -> the tie-break rule is a real cost and the
#                                 one-line change is the fix
#   zqTR ~ zqE (0-1)           -> the tie-break is a symptom of a flat level 2,
#                                 not the cause; the fix has to make L2 learn
#   zqTM >> zqTR               -> the damage is the variance, not the bias:
#                                 randomising is the wrong repair, centring is
#   zqEAT > zqEA               -> the effect is additive with refine_target=percell
#
# Default tie_break=argmax is byte-identical to the shipped path (self-tested), so
# no existing arm or snapshot changes.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"

ppu=0
arm() {
  local name="$1" seed="$2"; shift 2
  # RAM *and* DISK. The launch gates in phases 8-10 checked only free RAM, and on
  # 2026-08-18 the filesystem reached 10T of 10T and stopped all sixteen live arms
  # within two minutes with no error in any log. A run dir is ~250 GB and is almost
  # entirely replay buffer.
  local free_gb avail_gb
  free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  avail_gb=$(df -BG --output=avail /mnt/workspace | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt 150 ]; then
    echo "SKIP $name: only ${free_gb} GB RAM available"; return 0
  fi
  if [ "${avail_gb:-0}" -lt 500 ]; then
    echo "SKIP $name: only ${avail_gb} GB DISK available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=100000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT $STOCK \
      in_train_eval=false save_eval_snapshot=false "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 5
}

for s in 1 2; do
  arm "p11_mp_zqTR_s${s}"  "$s" zoomq.tie_break=random
  arm "p11_mp_zqTM_s${s}"  "$s" zoomq.tie_break=middle
  arm "p11_mp_zqEAT_s${s}" "$s" zoomq.tie_break=random \
      zoomq.exec_cond_skeleton=true zoomq.refine_target=percell
done

echo "phase 11 launched: 6 arms on move_plate"
