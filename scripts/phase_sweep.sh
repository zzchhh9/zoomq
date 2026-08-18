#!/bin/bash
# Does a skeleton's success depend on WHERE the chunk grid happens to land?
#
# The error-budget sweep left one contradiction unresolved. Gate 1's original
# move_plate table (n=60) reads j=2 0.467 / j=3 0.817 / j=16 0.750; re-running
# the same projection with --keep-first (the off-by-one fix, which RAISES the
# unprojected ceiling to 0.800/0.867) reads j=2 0.100 / j=3 0.300 / j=16 0.800.
# j=16 agrees, the skeletons do not, and the sign flips -- so it is not "one
# action short is worse", it is alignment: keep_first shifts the whole action
# array by one and project_sequence cuts chunks at 0,16,32..., so the knots land
# on different timesteps.
#
# --phase slides the grid without changing which actions execute or how many.
# phase=1 should reproduce the old number if the effect is purely alignment.
#
# A deployed agent cannot choose its phase: the grid is set by when the episode
# starts and when it replans. So the spread across phase is a lower bound on how
# much of any skeleton result is luck rather than representation.
set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -eu
cd /mnt/workspace/zoomq
export HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
P=/mnt/workspace/anchorq/.venv/bin/python

run() {  # run <j> <phase>
  local j="$1" ph="$2" out="gates/phase_j${1}_${2}.json"
  [ -f "$out" ] && return 0
  $P gates/gate1_replay.py --task move_plate --era floating --hold 0 \
     --keep-first --mode full --j "$j" --num-demos 60 --workers 6 --phase "$ph" \
     --cache-root /mnt/workspace/zoomq/demos/.bigym \
     --out "$out" >> logs/phase_sweep.log 2>&1
  echo "j=$j phase=$ph done" >> logs/phase_sweep.log
}

# phase 0 and 1 first -- they are the two that resolve the contradiction.
run 3 0; run 3 1; run 2 0; run 2 1
for ph in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do run 3 "$ph"; done
for ph in 4 8 12; do run 2 "$ph"; done
echo "PHASE SWEEP COMPLETE" >> logs/phase_sweep.log
