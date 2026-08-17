#!/bin/bash
# Phase 7 — does conditioning Q^exec on the skeleton bring the mechanism to life?
#
#   ./scripts/launch_phase7.sh [--dry-run]
#
# THE QUESTION. ZoomQ's contribution is a stopping rule that reads Q^exec_r, "the
# value of stopping at round r and executing that skeleton". Shipped, that head
# could not see the skeleton: measured with autograd on a trained snapshot,
# dQ^exec_0/d(action) was EXACTLY 0.0, and Q^exec moved 0.064 C51 atoms across six
# wildly different chunks against 3.039 for the same network's per-cell head. Round
# 0 is the depth eval executes essentially always, so the rule was inert
# (Delta_r/kappa*u_r p90 = 0.103 over 45 arms) and Gate 2 could not be run honestly:
# it would have killed the adaptive headline for an implementation defect rather
# than for the method.
#
# `zoomq.exec_cond_skeleton=true` feeds the round-r skeleton to the head. It is the
# branch main.tex App. C says an executable-coarse-chunk reading REQUIRES, and that
# zoomq.tex 1 says ZoomQ takes. Verified structurally in zoomq_probe.py T13:
# dQ^exec_r/da becomes nonzero at EVERY round including 0, and with the flag off it
# is still bit-exactly 0 (so arms trained before this change remain comparable).
#
# WHY move_plate ONLY. It is the sole task whose CQN-AS baseline is reproduced here
# (p1_mp_s1/s2/s3 at 0.80/0.68/0.68 by 50K against the published 64.0 +- 7.5), so it
# is the only place a ZoomQ number is unambiguous. On take_cups/put_cups BOTH
# methods sit on the floor at 20K -- the baseline itself reads 0.08 then 0.00 at
# consecutive snapshots -- and nothing is resolvable there against a +-6pp
# 25-episode noise floor.
#
# THE ARMS (12). One variable at a time, four seeds each.
#   zqA   ZoomQ exactly as it shipped. The control, re-run fresh so it is matched
#         to zqS in seeds and schedule rather than compared across waves.
#   zqS   + exec_cond_skeleton=true. THE fix, and the only difference from zqA.
#   zqSF  + exec_match_mode=online_minimal + exec_loss_norm=per_depth on top.
#         Those two were measured NOT to help on their own (exec AUC stayed
#         0.505-0.530) because they fix depth attribution, not action blindness.
#         Stacked on the fix they may finally bite: attribution only matters once
#         the heads can represent different values in the first place.
#
# EVAL_EVERY=2000, not 10000. The decisive evidence is a MECHANISM counter that a
# CPU snapshot probe reads in minutes -- Q^exec's action sensitivity, its
# demo-vs-random AUC, the across-depth spread, Delta/kappa*u. None of that needs a
# converged policy, so the first snapshot is the first measurement, and waiting 10K
# frames for it would delay the answer fivefold for nothing. Success rate is still
# read later from the same arms.
#
# SIZING. Measured just before launch: 134.7 of 184 cores idle (the co-tenant
# diffusionsafeguards job finished, dropping from 110 cores to 2) and 1236 GB RAM
# free. At the measured ~56 GB/arm at 100K frames, 12 arms need ~670 GB -- inside
# the budget with room, and RAM is the binding constraint here, not cores
# (see the ali-ram-caps-bigym-arms note: 13 arms were lost to the OOM killer when
# an earlier wave was sized by core count).
# WORKERS=8 is the py-spy optimum on an uncontended box (49% env step + render,
# 16% in the replay DataLoader at the default 2 workers); the box is 73% idle now,
# so it applies again.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
FIX="zoomq.exec_cond_skeleton=true"
ATTR="zoomq.exec_match_mode=online_minimal zoomq.exec_loss_norm=per_depth"

ppu=0
arm() {
  local name="$1" seed="$2"; shift 2
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${free_gb:-0}" -lt 120 ]; then
    echo "SKIP $name: only ${free_gb} GB available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 4
}

for s in 1 2 3 4; do
  arm "p7_mp_zqA_s${s}"  "$s"
  arm "p7_mp_zqS_s${s}"  "$s" $FIX
  arm "p7_mp_zqSF_s${s}" "$s" $FIX $ATTR
done

echo "phase 7 launched: 12 arms on move_plate"
