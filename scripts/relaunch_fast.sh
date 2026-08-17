#!/bin/bash
# Restart the six 0-frame move_plate mechanism arms with settings that actually
# fit the box. They lose nothing: none has written a frame.
#
#   ./scripts/relaunch_fast.sh [--dry-run]
#
# WHY. `ps -eo pcpu` reports a process's LIFETIME AVERAGE, not its instantaneous
# share, which is why the box looked like it was using 77 of 184 cores while
# every arm ran at 0.21 fps. vmstat tells the truth: run queue 271-356 against
# 184 cores, 80% user, 1% idle, no swap, only 3 processes in D state. The box is
# ~2x oversubscribed on CPU, and the cause is WORKERS=8 x 42 arms = 336
# dataloader workers. CLAUDE.md 2.5's "be generous with workers on a 184-core
# box" was written for a handful of arms, not forty.
#
# WORKERS=2. At 42 arms, 8 workers per arm is 336 processes competing for 184
# cores; the py-spy profile that justified 8 (16% of time in the replay
# DataLoader, pin_memory 10.4%) was taken on an UNCONTENDED box. Under 2x
# oversubscription extra workers cost more in context switching than they save.
#
# EVAL_EVERY=2000 instead of 10000. The first snapshot is the first moment any of
# these arms can be measured, and the question they answer is a MECHANISM question
# (is Q^exec a function of the action?) that the snapshot probe can settle at any
# frame count -- it does not need a converged policy. Waiting 10K frames for the
# first snapshot delays the answer by 5x for no benefit. Success rate still gets
# read later from the same arms.
#
# These six are the whole live question. move_plate is the only task whose CQN-AS
# baseline is reproduced here (0.80/0.68/0.68 at 50K vs published 64.0 +- 7.5),
# and on take_cups/put_cups BOTH methods are still on the floor at 20K -- the
# baseline itself reads 0.08 then 0.00 at consecutive snapshots, so nothing is
# resolvable there yet.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"

cd /mnt/workspace/zoomq
ARMS=(p5_mp_zqD_s1 p5_mp_zqD_s2 p6_mp_zqP_s1 p6_mp_zqP_s2 p6_mp_zqW_s2 p6_mp_zqW_s3)

# Refuse to restart anything that has made progress -- this script is only for
# arms that are still at zero and are therefore free to relaunch.
for a in "${ARMS[@]}"; do
  fr=$(tail -1 "runs/$a/train.csv" 2>/dev/null | cut -d, -f1 | cut -d. -f1)
  if [ -n "${fr:-}" ] && [ "${fr:-0}" -gt 100 ] 2>/dev/null; then
    echo "ABORT: $a is at frame $fr, not zero. Refusing to restart it."
    exit 1
  fi
done

if [ -z "$DRY" ]; then
  for a in "${ARMS[@]}"; do
    pids=$(pgrep -f "/runs/${a}\b" || true)
    [ -n "$pids" ] && kill $pids 2>/dev/null
  done
  sleep 15
  for a in "${ARMS[@]}"; do
    pids=$(pgrep -f "/runs/${a}\b" || true)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  done
  sleep 5
fi

L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos
FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"

ppu=0
arm() {
  local name="$1" seed="$2"; shift 2
  $DRY env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=2 \
    setsid nohup $L "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 10 ))
  sleep 6
}

arm p5_mp_zqD_s1 1 zoomq.adaptive=false
arm p5_mp_zqD_s2 2 zoomq.adaptive=false
arm p6_mp_zqP_s1 1 zoomq.cond_mode=final
arm p6_mp_zqP_s2 2 zoomq.cond_mode=final
arm p6_mp_zqW_s2 2 zoomq.w_scale=1e6
arm p6_mp_zqW_s3 3 zoomq.w_scale=1e6

echo "relaunched 6 move_plate mechanism arms at WORKERS=2 EVAL_EVERY=2000"
