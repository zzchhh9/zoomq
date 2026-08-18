#!/bin/bash
# Phase 12 — is the LATTICE the binding constraint? Widen the deep-round windows.
#
#   ./scripts/launch_phase12.sh [--dry-run]
#
# WHY, and why not the tie-break. Phase 11 settled that line and it is closed:
#   zqE           1, 0        (no fix)
#   zqE + random  0, 0  @45   and 0, 0 @90   -- WORSE
#   zqE + middle  0, 0  @45   and 0, 0 @90   -- WORSE, despite a BETTER frozen-weight
#                                               chunk MAE (0.0611 vs 0.0713)
#   zqEA          4, 7  @45   12 @90
#   zqEA + random 4, 1  @45   11, 8 @90      -- inside noise
# Exactly as the frozen-weight counterfactual predicted (zqE MAE 0.07127 -> 0.07595
# under random, 8/8 seeds worse, 9.48 sigma). The tie-break re-labels flat cells; it
# cannot un-flatten them.
#
# WHAT THE PROBES SAY IS ACTUALLY WRONG. The BC loss is NOT blocked from the finest
# amplitude level -- its per-sample head gradient there is 2.77e-4 against 4.93e-5 at
# L0, i.e. 5.6x LARGER; there is no level mask (bc_dim_weights is all-ones); the
# dueling mean-subtraction leaves the bin spread bit-identical; and TD is 0.25% of BC
# at the head with cosine +0.015, so it is not fighting it. The target itself is
# unlearnable: swapping ONLY the target bin for a constant opens L2 from 0.0019 to
# 6.24 atoms in 25 steps at the same weights, lr and Adam state.
#
# The target is unlearnable because of lattice geometry. Window half-widths by round
# are 1.0/0.15/0.08/0.04/0.02, so the L2 bin is 0.016/0.0024/0.00128/0.00064/0.00032
# against CQN-AS's 0.016 everywhere: 87.5% of cells sit on an L2 bin 12-50x finer
# than the baseline's, while ZoomQ's own chunk error is 0.0215 -- up to 70x coarser
# than the bin it is being asked to resolve. The finest level is being trained on
# noise, and learning nothing from noise is correct behaviour.
#
# Two independent confirmations already on disk:
#   * A 400-step continuation with w_of_t forced to 1.0 moves L2 bin accuracy
#     0.339 -> 0.532 and L1 0.457 -> 0.866 (out-of-distribution, not a retrain).
#   * At ROLLOUT the demo knot falls outside the policy's OWN window in 23.5% of
#     round-3 cells and 47.5% of round-4 cells for zqEA (0.7%/1.4% teacher-forced),
#     and round 4 owns 7 of the 16 timesteps -- 44% of the chunk. Widening addresses
#     that exposure-bias gap at the same time.
#
# THE ARMS (2), one treatment, two seeds. Config only; nothing under third_party.
#   zqEAw = zqEA + zoomq.w_schedule=[1.0,0.2,0.2,0.2,0.2]
# Round 0 is unchanged at 1.0. Rounds 1-4 go to a uniform 0.2 half-width, so the L2
# bin becomes 0.0032 everywhere (10x coarser at round 4), and the window sits
# 10.6/37/105/185x wider than the measured per-round residual medians of
# 0.0189/0.0054/0.0019/0.0011.
#
# PRE-REGISTERED DECISION RULE. Readout: successful training rollouts in the first
# 45 episodes from train.csv episode_reward -- the readout that produced 23/21/16
# and 4,7. The comparison baseline is zqEA's OWN 4, 7 = 11/90, NOT zqE's 1/90.
#   pooled >= 20 / 90  (Poisson lambda=11 -> p ~ 0.007)
#        -> the lattice was the binding constraint; escalate to 100K and sweep the
#           per-round window schedule
#   pooled <= 11 / 90
#        -> null; window resolution is not it. Go after L1's target quality directly
#           (zqE L1 bin accuracy 0.512, L1 margin loss 0.0635 against the fully-flat
#           plateau of 0.1) or the exposure-bias path
#   12 - 19  -> inconclusive; do not build on it
# Only ONE treatment gets the budget: with a 0/45-vs-7/45 noise floor, one seed of
# each of two treatments is unreadable.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"
AB="zoomq.exec_cond_skeleton=true zoomq.refine_target=percell"

ppu=6
arm() {
  local name="$1" seed="$2"; shift 2
  local free_gb avail_gb
  free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  avail_gb=$(df -BG --output=avail /mnt/workspace | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt 150 ]; then echo "SKIP $name: ${free_gb} GB RAM"; return 0; fi
  if [ "${avail_gb:-0}" -lt 500 ]; then echo "SKIP $name: ${avail_gb} GB DISK"; return 0; fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=100000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT $STOCK $AB \
      in_train_eval=false save_eval_snapshot=false "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 5
}

for s in 1 2; do
  arm "p12_mp_zqEAw_s${s}" "$s" 'zoomq.w_schedule=[1.0,0.2,0.2,0.2,0.2]'
done

echo "phase 12 launched: 2 arms on move_plate"
