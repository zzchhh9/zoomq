#!/bin/bash
# Restart the 8 highest-value arms lost to the OOM episode. Nothing else.
#
#   ./scripts/relaunch_priority.sh [--dry-run]
#
# WHAT HAPPENED. Phases 5 and 6 were launched into a box that was already at
# 1720 of 1800 GB, so 14 of their arms died before writing a single frame, and
# three arms with real progress went with them. Measured PSS per arm is 17.9 GB
# below 5K frames and 33.0 GB above 40K, i.e. ~56 GB to reach 100K. Against
# ~1600 GB of usable RAM that caps the box at roughly 28 arms at 100K frames --
# far below the 65 that had been defined. scripts/cull.sh has since stopped 14
# arms the user approved, freeing 282 GB.
#
# SO THIS RESTARTS 8, NOT 17, AND CONCENTRATES THEM ON move_plate.
# That is not only a budget decision, it is a better design. move_plate is the
# ONLY task whose CQN-AS baseline is reproduced here (0.68-0.80 at 50K across
# three seeds, against the published 64.0 +- 7.5), so a ZoomQ number there is
# unambiguous. On take_cups the baseline is 0.08 and the 25-episode noise floor
# is +-6pp, so nothing is resolvable at that scale however many arms are run.
#
# THE EIGHT
#   p6_mp_zqP_s1/s2   cond_mode=final. Tests the prime suspect: Q^exec is not a
#                     function of the action (AUC 0.50-0.54 vs random chunks at
#                     every round) and it is the SOLE bootstrap carrier. Under
#                     cond_mode=midpoint a round-0 boundary sees p=0 (root_mask)
#                     and w=1.0, i.e. nothing about which values were chosen;
#                     cond_mode=final re-feeds earlier rounds at their committed
#                     values and is the only config-reachable way to open that
#                     channel.
#   p6_mp_zqW_s2/s3   w_scale=1e6. Reproduces arm C, which repairs the per-bin
#                     critic completely (tie fraction 0.000, demo-vs-shift
#                     +2.87 dz against the baseline's +1.21) and STILL scores
#                     0.00. That dissociation is load-bearing for the whole
#                     diagnosis and is currently n=1 at a single snapshot.
#   p5_mp_zqD_s1/s2   adaptive=false. Executes the full-depth chunk, i.e. the
#                     object the bootstrap already values; kills the
#                     execution-vs-valuation mismatch outright.
#   p2_pc_zqF_s1/s2   Resumed from snapshot at 46,249 / 45,742 frames. Not a new
#                     question -- simply the largest sunk cost in the OOM
#                     episode, and they resume rather than restart.
#
# DEFERRED, NOT ABANDONED (would need ~110 GB more than is safe right now):
#   p5_tc_zqD_s1/s2, p5_pc_zqD_s1, p5_mp_zqK_s1, p5_tc_zqK_s1, p4_pc_zqE_s1/s2,
#   p3_pc_ctl_s1. The zqK arms in particular are subsumed by zqD: if executing at
#   full depth fixes nothing, re-tuning kappa so the stopping rule can fire is
#   pointless.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"

ppu=0
next_ppu() { ppu=$(( (ppu + 1) % 10 )); }

arm() {
  local name="$1" task="$2" seed="$3"; shift 3
  # Refuse to start anything while the box is tight; each start costs ~18 GB.
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${free_gb:-0}" -lt 60 ]; then
    echo "SKIP $name: only ${free_gb} GB available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=10000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task="$task" $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      >> "logs/${name}.log" 2>&1 < /dev/null &
  next_ppu
  sleep 8
}

arm p6_mp_zqP_s1 move_plate 1 zoomq.cond_mode=final
arm p6_mp_zqP_s2 move_plate 2 zoomq.cond_mode=final
arm p6_mp_zqW_s2 move_plate 2 zoomq.w_scale=1e6
arm p6_mp_zqW_s3 move_plate 3 zoomq.w_scale=1e6
arm p5_mp_zqD_s1 move_plate 1 zoomq.adaptive=false
arm p5_mp_zqD_s2 move_plate 2 zoomq.adaptive=false
# These two resume from snapshot.pt rather than starting over.
arm p2_pc_zqF_s1 put_cups 1 zoomq.exec_match_mode=online_minimal zoomq.exec_loss_norm=per_depth
arm p2_pc_zqF_s2 put_cups 2 zoomq.exec_match_mode=online_minimal zoomq.exec_loss_norm=per_depth

echo "relaunched up to 8 priority arms"
