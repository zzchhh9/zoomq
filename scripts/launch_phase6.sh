#!/bin/bash
# Phase 6 — the exec head is action-blind. Test the one channel that could fix it.
#
#   ./scripts/launch_phase6.sh [--dry-run]
#
# WHAT THE SNAPSHOT PROBE MEASURED
# --------------------------------
# On matched ~40K-frame snapshots, 384 real demo states, everything in action
# space or Q-value space, atom spacing dz = 0.08:
#
#   Q^exec is NOT A FUNCTION OF THE ACTION. AUC 0.50-0.54 against uniform random
#   chunks at EVERY round; cross-round spread 0.0021 raw = 0.026 dz. The
#   `online_minimal` + `per_depth` exec fix does not change it (AUC 0.505-0.530).
#
# That is fatal in combination with the backup audit's finding that Q^exec is the
# SOLE bootstrap carrier: `nxt["exec_probs"]` (zoomq.py:1145-1148) is the only
# thing read from the next state, and every per-cell refine target chains down to
# it. So ZoomQ's entire temporal credit assignment flows through a head that
# carries no action information.
#
# The inherited per-cell critic is alive but blunt, and blind over time:
#   demo vs uniform random    ZoomQ +0.34/+0.71 dz   baselines +1.88/+1.43 dz
#   demo vs SAME EPISODE +-24 steps
#                             ZoomQ +0.01/+0.05 dz   baselines +1.37/+1.08 dz
#                             AUC     0.537/0.542              0.87/0.87
#   per-cell Q std within one observation
#                             ZoomQ 0.026/0.113 dz   baselines 2.23/2.12/2.24 dz
#   action-range / state-std  ZoomQ 0.32/0.50        baselines 1.39-3.14
# i.e. ZoomQ's Q is dominated by the STATE where the baseline's is dominated by
# the ACTION, a 6-10x inversion, and ZoomQ is at chance at telling apart two
# actions 24 steps apart in the same episode.
#
# TWO THINGS THE PROBE REFUTED, so do not spend arms on them
#   - Tie-break / rail saturation. 66-69% of finest-level cells do have spread
#     &lt; 1e-4 (median exactly 0.000), but selected bins come out near-uniform
#     across all five and the decoded action shows no rail pile-up (5.7% at
#     -0.992 vs the demos' own 6.1%) and no bias (+0.0009). Near-flatness is not
#     a bit-exact tie, so `qs.max(-1)[1]` still dithers on ~1e-5 noise. The
#     missing `random_action_if_within_delta` costs nothing measurable.
#   - Action collapse. Selected-chunk diversity matches the demos to within 2%,
#     R^2 0.95, error uniform across all 16 dims.
#
# THE DISSOCIATION THAT NARROWS IT (arm C, w_scale=1e6 -&gt; every residual window
# is the full [-1,1]; everything else ZoomQ). On move_plate against mp_base:
#   per-bin Q fully RECOVERS and exceeds the baseline: tie fraction 0.000 at
#   every round and level, demo-vs-shift +2.87 dz / win 1.000 (mp_base +1.21 dz),
#   executes at full depth, knot MAE 0.0126 vs the baseline's 0.0108, R^2 0.986.
#   AND IT STILL SCORES 0.00, every rollout at exactly 300 steps.
# So flat per-bin Q, ties and action error are NOT sufficient. What arm C still
# shares with the failing arms is the dead Q^exec head.
#
# WHY THE EXEC HEAD IS ACTION-BLIND, AND THE ARM THAT TESTS IT
# ------------------------------------------------------------
# `exec_logits = exec_head(finest.index_select(1, boundary_pos))` (zoomq.py:402-405)
# reads the GRU features at each round boundary. Under `cond_mode: midpoint` --
# the running config on 53 of 55 arms -- each decision position is conditioned on
# its OWN level-l interval midpoint plus the parent prediction p and window w.
# At round 0 there are no committed neighbours: `parent_prediction` is zeroed by
# `root_mask` and w = 1.0 for both knots, so the round-0 boundary state sees
# NOTHING about which values were chosen. That is the same missing channel the
# probe recorded as exactly 0.000e+00 in T10 (zoomq-coordination-channel).
#
#   zqP  cond_mode=final
#        Earlier rounds are re-fed at their FINAL COMMITTED VALUES via a second
#        prefix pass per round, and it is that pass's hidden state that is
#        carried forward (zoomq.py:599-653 / 805-825). This is the ONLY mechanism
#        in the code that puts a chosen action value into the level-0 recurrence,
#        so it is the only config-reachable way Q^exec can become a function of
#        the action. Costs 2x the recurrent work; verified as exactly 2x the
#        launches and GRU steps.
#
#   zqW  w_scale=1e6, 2 seeds on move_plate
#        Reproduce arm C. The probe's own falsifier #2 is that arm C is n=1, one
#        snapshot at 35,880 frames with its only logged eval at 25,080. The
#        dissociation it carries is load-bearing for the whole diagnosis, so it
#        needs a second and third seed before anything is built on it.
#
# PRE-REGISTERED DECISION RULE (25-episode offline eval; the n=25 noise floor on
# move_plate is +-6pp, measured: the same weights score 0.68-0.80 across seeds)
#   zqP makes Q^exec action-dependent (AUC well above 0.55 on the same probe)
#       AND SR &gt; 0  -&gt; the action-blind bootstrap is the root cause. Confirmed.
#   zqP makes Q^exec action-dependent but SR stays 0.00
#       -&gt; the bootstrap is not sufficient either; the remaining suspect is the
#          dyadic decomposition itself, and the honest paper result is negative.
#   zqP leaves Q^exec flat
#       -&gt; the blindness is not the conditioning; look at the exec head's own
#          target (it is one per-sample distribution broadcast across depths, so
#          it may simply have no action-contrastive signal to learn from).
#   zqW disagrees with arm C -&gt; the dissociation was a single-seed artefact and
#       the per-bin flatness is back on the table.
#
# All config-only. bigym_src/zoomq.py is untouched, so the 59 arms already
# running are unaffected even when the watchdog restarts them.
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
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=10000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task="$task" $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  next_ppu
  sleep 2
}

for s in 1 2; do
  arm "p6_mp_zqP_s${s}" move_plate "$s" zoomq.cond_mode=final
  arm "p6_tc_zqP_s${s}" take_cups  "$s" zoomq.cond_mode=final
done

for s in 2 3; do
  arm "p6_mp_zqW_s${s}" move_plate "$s" zoomq.w_scale=1e6
done

echo "phase 6 launched: 6 arms"
