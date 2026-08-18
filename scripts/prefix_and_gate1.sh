#!/bin/bash
# 1) THE EXACT-PREFIX TEST -- isolate the phase sweep's cause.
#    The grid stays at phase 0 (what a deployed agent actually gets: it replans
#    from t=0, so every chunk is a full sparse K=16). Only the first N actions
#    are executed verbatim. If success climbs steeply for small N, the whole
#    phase effect is the opening steps and nothing periodic.
#
# 2) THE CORRECTED GATE 1 -- the original move_plate table was measured under
#    the off-by-one replay (keep_first=False), which shifts the chunk grid one
#    step and lands in the good band. Under the corrected replay at the
#    deployment phase, j=2 reads 0.100 and j=3 reads 0.367 against an
#    unprojected 0.800. Gate 1 asks: does ANY j<16 reach 90% of the j=16
#    ceiling? j=5 and j=9 were never measured this way.
set +eu; source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1; set -eu
cd /mnt/workspace/zoomq
export HOME=/mnt/workspace/zoomq/demos PYTHONUSERBASE=/root/.local
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
P=/mnt/workspace/anchorq/.venv/bin/python
run() {  # run <out> <extra args...>
  local out="$1"; shift
  [ -f "$out" ] && return 0
  $P gates/gate1_replay.py --task move_plate --era floating --hold 0 \
     --keep-first --mode full --num-demos 60 --workers 6 \
     --cache-root /mnt/workspace/zoomq/demos/.bigym \
     --out "$out" "$@" >> logs/prefix_gate1.log 2>&1
  echo "wrote $out" >> logs/prefix_gate1.log
}
for n in 2 5 10 16 32 48; do
  run "gates/prefix_j3_${n}.json" --j 3 --phase 0 --exact-prefix "$n"
done
run "gates/gate1fix_all.json" --j 2,3,5,9,16 --phase 0
echo "PREFIX+GATE1 COMPLETE" >> logs/prefix_gate1.log
