#!/bin/bash
# Speed up the decisive wave by removing CPU contention. USER-APPROVED.
#
#   ./scripts/speedup.sh --dry-run
#   ./scripts/speedup.sh --confirm
#
# WHY, MEASURED. py-spy on a live arm (30 s @ 60 Hz) attributes wall clock as:
#     camera render      55.4%   (3x 84x84, CPU/EGL -- cannot be removed, BiGym
#                                 pixel observations must be rendered)
#     replay sampling    26.8%   <- this is the lever
#     gradient update     8.8%   (GPU)
#     other               8.8%
#     MuJoCo physics      0.2%
# Time-averaged PPU utilisation is ~32% and bursty (0,100,0,100 across samples):
# the batch-256 update finishes in a flash and then waits on the CPU. So raising
# GPU load cannot help -- its whole share is 8.8%, and that is the ceiling on any
# GPU-side speedup.
#
# The 26.8% in replay sampling is anomalous. An earlier profile of the same code
# measured 16% (pin_memory 10.4%) at the DEFAULT 2 workers. The difference is
# contention: 34 arms x WORKERS=8 = 272 dataloader processes competing for 184
# cores, so the workers now cost more in scheduling than they save in prefetch.
# CLAUDE.md 2.5's "be generous with workers on a 184-core box" was written for a
# handful of arms.
#
# TWO ACTIONS
#
# 1. Restart the phase-9 arms at WORKERS=3. They are the only wave that can still
#    answer anything (ZoomQ under the STOCK protocol, where the baseline scores
#    16-23 successes per 45 episodes) and they are at ~0 frames, so restarting
#    costs nothing. Refuses to restart any arm past RESTART_MAX_FRAMES.
#
# 2. Stop the baselines that have already reached 100K frames. They are finished:
#    every snapshot, train.csv row and eval result is on disk, and they produce no
#    further information while holding cores and render bandwidth. Reversible --
#    add the name back to watchdog.sh's KEEP list to resume from snapshot.
#
# Expected: ~272 -> ~90 dataloader workers, and 8 fewer arms rendering. The
# decisive counter (successful training rollouts in the first 45 episodes) should
# arrive in roughly an hour instead of 2.5.
set -uo pipefail

MODE="${1:---dry-run}"
[ "$MODE" = "--confirm" ] || MODE="--dry-run"
RESTART_MAX_FRAMES="${RESTART_MAX_FRAMES:-3000}"
DONE_FRAMES="${DONE_FRAMES:-99000}"

cd /mnt/workspace/zoomq
PY=/mnt/workspace/anchorq/.venv/bin/python
frames_of() {
  $PY -c "
import csv, os, sys
p = '/mnt/workspace/zoomq/runs/%s/train.csv' % sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
rs = list(csv.DictReader(open(p)))
print(int(float(rs[-1]['frame'])) if rs else 0)
" "$1" 2>/dev/null || echo 0
}

echo "### 1. finished baselines to stop (>= ${DONE_FRAMES} frames)"
STOPPED=()
for a in p1_mp_s1 p1_mp_s2 p1_mp_s3 p1_sh_s1 p1_sh_s2 p1_sh_s3 \
         p2_tc_base_s1 p2_tc_base_s2 p2_pc_base_s1 p2_pc_base_s2 \
         p2_fc_base_s1 p2_fc_base_s2; do
  pgrep -f "train_cqn_as_bigym.*runs/${a}\b" >/dev/null 2>&1 || continue
  f=$(frames_of "$a")
  if [ "${f:-0}" -ge "$DONE_FRAMES" ]; then
    echo "  [$MODE] stop $a (${f} frames -- finished)"
    STOPPED+=("$a")
    if [ "$MODE" = "--confirm" ]; then
      pids=$(pgrep -f "train_cqn_as_bigym.*runs/${a}\b" || true)
      [ -n "$pids" ] && kill $pids 2>/dev/null
    fi
  else
    echo "  keep  $a (${f} frames -- still short of ${DONE_FRAMES})"
  fi
done

echo
echo "### 2. phase-9 arms to restart at WORKERS=3"
P9=()
for s in 1 2; do for k in zqE zqES zqEB zqEA; do
  a="p9_mp_${k}_s${s}"
  f=$(frames_of "$a")
  if [ "${f:-0}" -gt "$RESTART_MAX_FRAMES" ]; then
    echo "  SKIP  $a (${f} frames > ${RESTART_MAX_FRAMES}; too much progress to discard)"
  else
    echo "  [$MODE] restart $a (${f} frames)"
    P9+=("$a")
  fi
done; done

[ "$MODE" != "--confirm" ] && { echo; echo "dry run only"; exit 0; }

for a in "${P9[@]}"; do
  pids=$(pgrep -f "train_cqn_as_bigym.*runs/${a}\b" || true)
  [ -n "$pids" ] && kill $pids 2>/dev/null
done
sleep 20
for a in "${P9[@]}"; do
  pids=$(pgrep -f "train_cqn_as_bigym.*runs/${a}\b" || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
done
sleep 8

export DEMO_HOME=/mnt/workspace/zoomq/demos
FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"
STOCK="temporal_ensemble=true nstep=1"
ppu=8
relaunch() {
  local name="$1" seed="$2"; shift 2
  env DEMO_HOME="$DEMO_HOME" SEED="$seed" EVAL_EVERY=2000 EVAL_EPS=25 WORKERS=${P9_WORKERS:-8} \
    setsid nohup ./scripts/launch_arm.sh "$name" "$ppu" config_zoomq_bigym \
      bigym_task=move_plate $FLOAT $STOCK \
      in_train_eval=false save_eval_snapshot=true "$@" \
      > "logs/${name}.log" 2>&1 < /dev/null &
  ppu=$(( (ppu + 1) % 16 ))
  sleep 3
}
SKEL="zoomq.exec_cond_skeleton=true"
PERCELL="zoomq.refine_target=percell"
for s in 1 2; do
  relaunch "p9_mp_zqE_s${s}"  "$s"
  relaunch "p9_mp_zqES_s${s}" "$s" $SKEL
  relaunch "p9_mp_zqEB_s${s}" "$s" $PERCELL
  relaunch "p9_mp_zqEA_s${s}" "$s" $SKEL $PERCELL
done

echo
echo "stopped ${#STOPPED[@]} finished baselines; relaunched ${#P9[@]} phase-9 arms at WORKERS=3"
