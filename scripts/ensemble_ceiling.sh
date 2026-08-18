#!/bin/bash
# The ceiling that actually applies to phase 9.
#
# Everything measured so far executes ONE chunk open loop -- ZoomQ's own regime
# (temporal_ensemble=false). Under it the round-0 skeleton tops out at 0.100 and
# round 1 at 0.367 against an unprojected 0.867. But phase 9 deliberately runs
# ZoomQ on the STOCK protocol, where the agent replans every step and
# TemporalEnsembleControl averages ~16 overlapping plans (gain 0.01 -> weights
# 1.00..0.86, i.e. near-uniform). Staggered knots across those 16 plans should
# recover most of what a single sparse skeleton loses -- but "should" is not a
# measurement, and if it does not, phase 9 cannot work no matter how the
# stopping rule behaves.
set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -eu
cd /mnt/workspace/zoomq
export HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
P=/mnt/workspace/anchorq/.venv/bin/python
$P gates/gate1_replay.py --task move_plate --era floating --hold 0 \
   --keep-first --mode ensemble --j 2,3,5,9,16 --num-demos 60 --workers 6 \
   --cache-root /mnt/workspace/zoomq/demos/.bigym \
   --out gates/ensemble_ceiling.json >> logs/ensemble.log 2>&1
echo "ENSEMBLE COMPLETE" >> logs/ensemble.log
