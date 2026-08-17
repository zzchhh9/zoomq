#!/bin/bash
# Phase 5 — test the root cause directly, config-only, no code change.
#
#   ./scripts/launch_phase5.sh [--dry-run]
#
# THE ROOT CAUSE TWO INDEPENDENT AUDITS CONVERGED ON
# --------------------------------------------------
# ZoomQ's critic never scores the action ZoomQ executes. Three separate
# mechanisms, all pointing the same way:
#
#   (a) The BOOTSTRAP values a full-depth chunk. `next_action` is built with
#       adaptive=False (zoomq.py:1141-1143), i.e. round 4, and then the target
#       takes a max over all five depths (:1148). The deployed policy executes
#       the ROUND-0 skeleton 91% of the time (depth_share_0 = 0.908).
#
#   (b) The per-cell refine target EXCLUDES Q^exec_r. zoomq.py:1167 is
#       `child = torch.stack([w_chain[r + 1] for r in range(R)], 1)`, so a cell
#       decided at round r is backed up to W_{r+1} -- the value of a plan with
#       at least one more refinement in it. `w_chain[0]`, computed at :1161-1163,
#       is never consumed: dead code, and the fingerprint of the missing branch.
#       Verified numerically: moving Q^exec_0 by +1.40 moves the round-0 cell
#       targets by exactly 0.000000e+00; the control (moving Q^exec_1) moves them
#       by 1.0. The diagonal of the sensitivity matrix is empty at every round.
#
#   (c) The STOPPING RULE that was meant to reconcile (a) and (b) cannot fire.
#       There is no twin critic here, so the paper's disagreement |Q_A - Q_B| was
#       substituted by the critic's own C51 spread (zoomq.py:893-896). A C51
#       spread has a FLOOR at the atom spacing; a disagreement does not. Measured:
#       exec_u_0 ~ 0.084 ~ 1.05 atoms, against a total across-depth spread of
#       Q^exec &lt;= 0.015. The rule needs a ratio &gt; 1 and its p90 never exceeds
#       0.103 on any of 45 arms. depth_share_1..4 ~ 0.020 each is exactly the
#       annealed eps-depth floor, i.e. the GREEDY depth is 0 in 100% of
#       un-forced rollouts, including on the `online_minimal` arms where the
#       exec-head tie WAS broken (exec_q rises 0.0385 -> 0.0536; still &lt; 0.088).
#
# So the agent commits 2 knots per dim -- 32 of 256 scalars -- under a criterion
# that presumes 14 more, then executes the straight line between them. Nothing in
# the model ever valued that line.
#
# THE ARMS (all config-only; bigym_src/zoomq.py is untouched, so the 49 arms
# already running are unaffected even when the watchdog restarts them)
# --------------------------------------------------------------------
#   zqD  zoomq.adaptive=false
#        The policy executes the FULL-DEPTH chunk -- exactly the object the
#        bootstrap already values. Kills (a) and (c) at once and makes (b)
#        harmless, because at full depth the executed chunk IS the refined plan.
#        This is the single most diagnostic arm available and it costs nothing.
#
#   zqK  zoomq.kappa=0.05
#        Attacks (c) alone. kappa=1.0 is mis-scaled by ~an order of magnitude for
#        BiGym's return scale: it must gate a &lt;=0.015 improvement with a yardstick
#        whose floor is 0.084.
#
#   zqU  zoomq.uncertainty=target_gap
#        Attacks (c) at the root: |Q_online - Q_target| has NO atom-spacing floor,
#        so unlike the C51 spread it can shrink below one atom as the critic
#        converges. The ablation the config file already documents.
#
# WHAT WOULD FALSIFY THE ROOT CAUSE
# ---------------------------------
#   zqD ~ 0.00 as well  -> execution/valuation mismatch is NOT the cause, and the
#                          fault is in what the critic learned, not in which
#                          action it commits to. That sends the investigation to
#                          the critic-discriminativeness probe.
#   zqD &gt;&gt; 0.00         -> confirmed. The fix is then a code change: point
#                          zoomq.py:1167 at `w_chain[r]`, and rescale the
#                          stopping rule.
#
# NOTE ON zqX (phase 4): those arms are NOT the ablation their header claims.
# The exec head is the SOLE bootstrap carrier -- `nxt["exec_probs"]` at
# zoomq.py:1145-1148 is the only thing read from the next state -- and the head
# is zero-initialised with `exec_lambda * exec_loss` as its only gradient path.
# So exec_lambda=0 pins it at zero forever and the critic degenerates into a
# 16-step Monte-Carlo reward regressor. Confirmed live: p4_tc_zqX_s1 has
# bellman_target_q == batch_reward == 0.0012367 to 5 s.f., exec_q_0 = 0,
# consistency_drift = 0, versus p4_tc_zqE_s1 (identical but exec_lambda=0.1) at
# bellman_target_q 0.0014445 &gt; batch_reward 0.001436. Read those two arms as
# "no bootstrap at all", not as "exec head removed".
#
# COST -- 10 arms x ~2.5 cores on top of ~88, i.e. ~113 of 184.
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

# move_plate leads: it is the only task whose baseline is reproduced
# (0.68-0.80 at 50K vs published 64.0 +- 7.5), so a ZoomQ number there is
# unambiguous, and its 300-step episodes make it the cheapest to evaluate.
for s in 1 2; do
  arm "p5_mp_zqD_s${s}" move_plate "$s" zoomq.adaptive=false
  arm "p5_tc_zqD_s${s}" take_cups  "$s" zoomq.adaptive=false
  arm "p5_pc_zqD_s${s}" put_cups   "$s" zoomq.adaptive=false
done

arm p5_mp_zqK_s1 move_plate 1 zoomq.kappa=0.05
arm p5_tc_zqK_s1 take_cups  1 zoomq.kappa=0.05
arm p5_mp_zqU_s1 move_plate 1 zoomq.uncertainty=target_gap
arm p5_tc_zqU_s1 take_cups  1 zoomq.uncertainty=target_gap

echo "phase 5 launched: 10 arms"
