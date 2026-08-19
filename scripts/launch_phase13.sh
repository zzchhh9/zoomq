#!/bin/bash
# Phase 13 — align acting with the Bellman next-action. ONE treatment, two seeds.
#
#   ./scripts/launch_phase13.sh [--dry-run]
#
# THE REASON, and it is not the replay-ceiling table.
# zoomq.py:1321 the bootstrap already descends with adaptive=False, i.e. the target
# values a FULL 16-knot chunk; zoomq.py:1272 act() descends with adaptive=True and,
# with the stopping rule dead (delta_over_kappa_u p90 ~1e-05 against the 1.0 it
# needs), returns round 0 on 79-89% of steps. So today the backup prices a 16-knot
# chunk while the body walks a 2-knot skeleton. adaptive=false makes them the same
# object. act() already computes all five rounds before selecting one, so FLOPs are
# unchanged -- this is a selection change, not extra compute.
#
# THE RISK, stated up front because it is real. Deep-round level 2 is flat (spread
# 0.0015-0.0024 atoms) and at rollout the demo knot falls outside the policy's own
# window in 23.5% of round-3 and 47.5% of round-4 cells. Forcing execution of those
# knots may be WORSE than executing round 0. This arm is worth running because it
# resolves a train/act mismatch, not because any ceiling table guarantees a win.
#
# WHAT IS NOT IN THIS WAVE, and why.
#   exec_lambda=0 -- rejected. `zq_exec_lambda * exec_loss` (zoomq.py:1459) is the
#     exec head's ONLY gradient; the BC terms act on q_probs_a and never touch it.
#     Zeroing it freezes the head at its zero-init, which makes exec_cond_skeleton
#     (Stage A, which only widens that head's input) a no-op. The arm would silently
#     be percell-only, i.e. zqEB, which scores 0,2 -- not "zqEA minus the exec tax".
#     To measure the tax properly, multiply exec_lambda by critic_lambda so that 0.1
#     is really 0.1, rather than driving it to 0. On `chain` arms zeroing it is worse
#     still: exec is the sole bootstrap carrier there, so the backup would collapse to
#     one-step MC (already observed on p4_tc_zqX, bellman_target_q == batch_reward).
#   an L2 mask -- rejected in this wave. Line A's masking works by NOT training
#     rounds 1-4's level 2; forcing round-4 execution means executing exactly those
#     untrained knots. The two interventions oppose each other. A per-round L2 head
#     belongs on an arm still executing round 0.
#   a second treatment -- rejected. Against a 0/45-vs-7/45 noise floor, one seed per
#     treatment is unreadable.
#
# save_eval_snapshot=true / EVAL_EVERY=2000 (in_train_eval=false, so no rollout cost)
# because phase 12 shipped with save_eval_snapshot=false and EVAL_EVERY=100000 and
# would have finished with no intermediate checkpoints -- spread_r0_L2 and
# binacc_r0_L2 are not train.csv columns and cannot be recovered post hoc.
#
# PRE-REGISTERED DECISION RULE. Successes in the first 45 episodes, pooled over both
# seeds, against zqEA's OWN 4 + 7 = 11/90:
#   >= 20 / 90  (Poisson lambda=11 -> p ~ 0.007)  the mismatch was real; escalate
#   <= 11 / 90                                    null
#   12 - 19                                       inconclusive; do not build on it
# READ ALSO, because the success count alone can mislead here:
#   depth_share_4  should rise to ~1 - 0.8*eps (greedy now sits at round 4). If it
#                  does not, the flag did not take.
#   spread_r0_L2 / binacc_r0_L2  from snapshot.pt via critic_localize.py. If successes
#                  reach 20+ while level 2 is still flat, the win came from using 16
#                  L0/L1 knots, NOT from level 2 -- and the asymptotic few points that
#                  a working level 2 would buy are still unavailable.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"
AB="zoomq.exec_cond_skeleton=true zoomq.refine_target=percell"

ppu=8
arm() {
  local name="$1" seed="$2"; shift 2
  local free_gb avail_gb
  free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  avail_gb=$(df -BG --output=avail /mnt/workspace | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt 150 ]; then echo "SKIP $name: ${free_gb} GB RAM"; return 0; fi
  if [ "${avail_gb:-0}" -lt 500 ]; then echo "SKIP $name: ${avail_gb} GB DISK"; return 0; fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT $STOCK $AB \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 5
}

for s in 1 2; do
  arm "p13_mp_zqEAd_s${s}" "$s" zoomq.adaptive=false
done

echo "phase 13 launched: 2 arms on move_plate (adaptive=false only)"
