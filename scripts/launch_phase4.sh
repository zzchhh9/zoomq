#!/bin/bash
# Phase 4 — complete the 2x2 factorial, and remove the exec head's gradient sink.
#
#   ./scripts/launch_phase4.sh [--dry-run]
#
# WHY THIS WAVE EXISTS
# --------------------
# Phase 2 compared `base` against `zqA` and found 0.08/0.12 vs 0.00/0.00. But
# those two arms differ in THREE variables at once:
#
#                       temporal_ensemble   nstep   algorithm
#   base                      true            1     CQN-AS
#   zqA / zqF                 false          16     ZoomQ
#
# `temporal_ensemble=true` is not a cosmetic difference. Per
# train_cqn_as_bigym.py:446-478 the baseline REPLANS EVERY CONTROL STEP and
# blends overlapping chunk predictions; the ZoomQ configs run
# `action[episode_step % action_sequence]`, i.e. open loop for 16 steps. That is
# a 16x difference in replanning frequency, and it has nothing to do with ZoomQ.
#
# Phase 3's `ctl` arm fills the bottom-left cell (CQN-AS under the matched
# protocol). This wave fills the top-right one:
#
#                       CQN-AS              ZoomQ
#   stock protocol      base  (0.08/0.12)   zqE   <- THIS WAVE
#   matched protocol    ctl   (phase 3)     zqA   (0.00/0.00)
#
# With all four cells the 0.12 -> 0.00 decomposes into a protocol main effect,
# an algorithm main effect, and their interaction. Without zqE the Phase 2
# comparison cannot be attributed at all.
#
# `temporal_ensemble` and `nstep` are trainer-level settings; `bigym_src/zoomq.py`
# never references either, and neither is in its `_UNSUPPORTED` list, so ZoomQ
# runs under the stock protocol unmodified. Expect a lower fps: with
# temporal_ensemble=true, act() is called every control step instead of every
# 16th, and ZoomQ's act() costs 5 rounds of C2F selection instead of 1.
#
# zqX — `zoomq.exec_lambda=0`, a ONE-VARIABLE delta from zqA.
#   ZoomQ's logged `td_loss_weighted` is `critic_lambda * refine_loss` only; the
#   exec term is omitted from it. The real non-BC critic objective is
#   0.118 (refine) + 0.1 * 1.212 (exec) = 0.239, so the EXEC HEAD IS ~50% OF THE
#   NON-BC CRITIC GRADIENT. That head is trained on targets that are provably
#   identical across depths (Q^exec_r spread 5e-4 over 5 depths, measured on 10
#   arms): the validity mask "the round-r skeleton reproduces the executed chunk"
#   selects exactly the cases where there IS no degradation, and by nestedness
#   the same label then feeds every deeper head. So half the critic gradient is a
#   constant-target regression pushed through the shared trunk that action
#   selection depends on. Since the stopping rule provably cannot fire, nothing
#   downstream uses the exec head, and exec_lambda=0 removes it cleanly.
#
# PRE-REGISTERED DECISION RULE (25-episode offline eval, >= 30K frames)
# --------------------------------------------------------------------
#   zqE ~ base                 -> the whole deficit is the execution protocol;
#                                 ZoomQ itself is not the problem, and every
#                                 Phase 2 number is void as an algorithm claim.
#   zqE ~ 0.00, ctl ~ base     -> the deficit is the ALGORITHM. Attributable.
#   zqE ~ 0.00, ctl ~ 0.00     -> both contribute; read the interaction.
#   zqX >> zqA                 -> the exec head's constant-target gradient was
#                                 poisoning the shared trunk.
#
# COST — 8 arms x ~2.5 cores = ~20 cores on top of ~155, i.e. ~175 of 184.
# This is deliberately at the top of the box (the user asked to maximise CPU
# use); the sim-work ceiling in CLAUDE.md 2.3 is ~150, so expect the existing
# arms to slow somewhat. Nothing is killed to make room.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
# The stock CQN-AS execution protocol, imposed on a ZoomQ agent.
STOCK="temporal_ensemble=true nstep=1"

ppu=0
next_ppu() { ppu=$(( (ppu + 1) % 10 )); }

arm() {
  local name="$1" cfg="$2" task="$3" seed="$4"; shift 4
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=10000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task="$task" $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  next_ppu
  sleep 2
}

# --------------------------------------- zqE: ZoomQ under the stock protocol --
# move_plate first: it is the only task whose baseline is reproduced
# (0.80/0.68/0.68 at 50K, mean 0.72, vs published 64.0 +- 7.5), so a ZoomQ
# number there is unambiguous.
for s in 1 2; do
  arm "p4_mp_zqE_s${s}" config_zoomq_bigym move_plate "$s" $STOCK
  arm "p4_tc_zqE_s${s}" config_zoomq_bigym take_cups  "$s" $STOCK
  arm "p4_pc_zqE_s${s}" config_zoomq_bigym put_cups   "$s" $STOCK
done

# ------------------------------------------- zqX: drop the exec head entirely --
arm p4_tc_zqX_s1 config_zoomq_bigym take_cups 1 zoomq.exec_lambda=0
arm p4_pc_zqX_s1 config_zoomq_bigym put_cups  1 zoomq.exec_lambda=0

echo "phase 4 launched: 8 arms"
