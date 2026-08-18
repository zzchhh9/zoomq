#!/bin/bash
# How much of the coarse-round deficit is bought back by replanning sooner?
#
# E=16 is ZoomQ today (commit the whole chunk open loop) -> round 0 caps at 0.100.
# E=1 + blending is the stock protocol -> round 0 caps at 0.617, but an agent
# replanning every step is not committing a skeleton and cannot be stopping early
# on one. E in between keeps the commitment and the depth decision. If the curve
# is already flat by E=4, the fix is a config knob (execution_length, which this
# repo already ships for ACT/DEAS/diffusion) plus ~4 lines in train_cqn_as_bigym.py,
# and ZoomQ's anytime story survives intact.
set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -eu
cd /mnt/workspace/zoomq
export HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
P=/mnt/workspace/anchorq/.venv/bin/python
for E in 1 2 4 8 16; do
  out=gates/execlen_${E}.json
  [ -f "$out" ] && continue
  $P gates/gate1_replay.py --task move_plate --era floating --hold 0 \
     --keep-first --mode execlen --j 2,3,5 --num-demos 60 --workers 6 \
     --exec-len "$E" --cache-root /mnt/workspace/zoomq/demos/.bigym \
     --out "$out" >> logs/execlen.log 2>&1
  echo "E=$E done" >> logs/execlen.log
done
echo "EXECLEN COMPLETE" >> logs/execlen.log
