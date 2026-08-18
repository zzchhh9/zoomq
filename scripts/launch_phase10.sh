#!/bin/bash
# Phase 10 — separate the two remaining questions, with the protocol confound gone.
#
#   ./scripts/launch_phase10.sh [--dry-run]
#
# WHERE THINGS STAND. One root cause is established with evidence at three
# independent levels, and one is not.
#
#   ESTABLISHED — committing 16 steps open loop.
#     ceiling  : a PERFECT round-0 policy caps at 0.100 vs an unprojected 0.867
#                (n=60 demo replay). Localised to the first ~5 control steps
#                after reset: executing just those verbatim takes j=3 from 0.367
#                to 0.900. Not knot precision -- at ZoomQ's measured knot MAE of
#                0.049, j=16 still scores 0.600.
#     control  : stock CQN-AS with ONLY temporal_ensemble=false nstep=16 flipped
#                goes from 80/94/105/112 successes per 160 episodes to 17 and 0.
#     repair   : execution_length=2 restores the round-0 ceiling to 0.833;
#                temporal ensembling restores it to 0.617.
#
#   NOT ESTABLISHED — whatever costs the remaining 0.617 -> 0.000. Phase 9 runs
#     on the stock protocol, whose round-0 ceiling is 0.617, and is 0/83 episodes
#     where the protocol-matched baseline gets 11/40. Under H0 that is p ~ 2.5e-12,
#     so a second defect exists. The best-evidenced candidate is the inert
#     stopping rule (delta_over_kappa_u p90 0.0147 against the 1.0 it needs;
#     depth_share bit-identical across arms sharing a seed, i.e. depth is 100%
#     eps), but inertness alone predicts "always executes round 0", which under
#     this protocol is worth 0.617, not 0. So it does not explain the zero either.
#
# THE ARMS (6), two seeds each.
#
#   zqFull  ZoomQ + zoomq.adaptive=false (max_round=4 is already the default, so
#           this descends to all 16 knots every time) + the STOCK protocol.
#           Representation identical to CQN-AS's, protocol identical to CQN-AS's;
#           the only remaining difference is ZoomQ's critic and backup. This is
#           the 2x2 cell arm C was meant to be before it ran under the protocol
#           that floors ctl.
#
#   zqE2    ZoomQ + execution_length=2 temporal_ensemble=false nstep=2.
#           Does the measured repair pay off in real training while KEEPING the
#           commitment, hence keeping the per-plan depth decision that the stock
#           protocol destroys? nstep follows E rather than staying at 16 on
#           purpose: the K-step backup is only attributable to one decision if it
#           spans what was actually committed, and at E=2 a 16-step reward sum
#           straddles eight different plans. It also avoids re-importing the
#           nstep=16 damage that ctl measures. The cost is that this arm differs
#           from stock ZoomQ in two ways at once -- acceptable for a does-the-fix-
#           work arm, which is why zqFull and ctl run beside it for attribution.
#
#   ctl     stock CQN-AS + temporal_ensemble=false nstep=16, seeds 3 and 4.
#           s1/s2 gave 0/160 and 17/160. n=2 cannot say how often the protocol
#           floors a seed outright, and that frequency is exactly what decides
#           whether any ZoomQ zero measured under it means anything.
#
# PRE-REGISTERED DECISION RULE — successful training rollouts, read from
# train.csv episode_reward, at matched episode counts. Stock-protocol baseline
# reference: 2/2/3/4 by episode 10, 16-29 by 45, 80-112 by 160.
#   zqFull > 0 by ~90 eps  -> the second defect is in the coarse rounds, and
#                             execution_length is the whole fix.
#   zqFull ~ 0 by ~160 eps -> ZoomQ's critic itself is broken independently of
#                             both protocol and representation. That is the clean
#                             negative result and it retires the anytime headline.
#   zqE2  ~ baseline       -> the fix works with commitment intact; this becomes
#                             the arm the paper reports.
#   ctl s3/s4 both > 0     -> 0/160 was an unlucky seed and ZoomQ's p7/p8 zeros
#                             are its own; either at 0 -> those zeros are
#                             uninterpretable and stay retired.
#
# COST. 6 arms. RAM measured at ~18 GB/arm below 5K frames and ~56 GB at 100K;
# 509 GB free after culling p7's twelve. The decision needs ~90-160 episodes
# (~27-48K frames), not 100K.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos

FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"
COMMIT16="temporal_ensemble=false nstep=16"

ppu=0
arm() {
  local name="$1" cfg="$2" seed="$3"; shift 3
  local free_gb; free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  if [ "${free_gb:-0}" -lt 150 ]; then
    echo "SKIP $name: only ${free_gb} GB available"; return 0
  fi
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=5000 EVAL_EPS=25 WORKERS=8 \
    setsid nohup $L "$name" "$ppu" "$cfg" \
      bigym_task=move_plate $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 5
}

for s in 1 2; do
  arm "p10_mp_zqFull_s${s}" config_zoomq_bigym "$s" $STOCK zoomq.adaptive=false
  arm "p10_mp_zqE2_s${s}"   config_zoomq_bigym "$s" \
      temporal_ensemble=false nstep=2 execution_length=2
done
for s in 3 4; do
  arm "p10_mp_ctl_s${s}"    -                  "$s" $COMMIT16
done

echo "phase 10 launched: 6 arms on move_plate"
