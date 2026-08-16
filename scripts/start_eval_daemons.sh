#!/bin/bash
# Automatic offline evaluation of the Phase 1/2 snapshots.
#
#   ./scripts/start_eval_daemons.sh
#
# The training arms run with `in_train_eval=false save_eval_snapshot=true`
# (CLAUDE.md's LAUNCH RULE), so they write a snapshot every 10k frames and pay
# no evaluation cost. This starts two `eval_daemon.py` instances that poll for
# new snapshots and score them on the otherwise-idle cores.
#
# --episodes 25 is CQN-AS's own reporting protocol ("success rate over 25
#   episodes"), which is what the numbers in main.tex Table 1 are.
# --videos 0 because a video eval has crashed a run here before with
#   EGLError (eglDestroyContext).
# Job counts are sized to leave the 30 training arms their ~89 cores: each eval
#   job is a full trainer process in eval_only mode.
#
# The daemon spawns children with the inherited environment, so HOME (which is
# what bigym derives its demonstration cache from) and PYTHONUSERBASE (which is
# where mujoco lives, and is also derived from HOME) must both be exported here
# or the children look in the wrong cache and fail to import.
set -euo pipefail

REPO=/mnt/workspace/zoomq/third_party/CQN-AS-G1
RUNS=/mnt/workspace/zoomq/runs
LOGS=/mnt/workspace/zoomq/logs

set +eu
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
set -eu

export MUJOCO_GL=egl
export HOME=/mnt/workspace/zoomq/demos
export PYTHONUSERBASE=/root/.local
export EVAL_PYTHON=/mnt/workspace/anchorq/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

cd "$REPO"

p1_paths=()
for a in p1_mp_s1 p1_mp_s2 p1_mp_s3 p1_sh_s1 p1_sh_s2 p1_sh_s3; do
  p1_paths+=(--path "$RUNS/$a")
done

p2_paths=()
for t in tc pc du fc; do
  for m in base zqA zqF; do
    for s in 1 2; do
      p2_paths+=(--path "$RUNS/p2_${t}_${m}_s${s}")
    done
  done
done

# Phase 1 is the gate, so it gets its own daemon and is never starved by the
# much longer Phase 2 episodes.
setsid nohup "$EVAL_PYTHON" scripts/eval_daemon.py \
  --gpu 0 --jobs 8 --episodes 25 --videos 0 --oldest-first \
  "${p1_paths[@]}" > "$LOGS/evald_p1.log" 2>&1 < /dev/null &

setsid nohup "$EVAL_PYTHON" scripts/eval_daemon.py \
  --gpu 1 --jobs 16 --episodes 25 --videos 0 --oldest-first \
  "${p2_paths[@]}" > "$LOGS/evald_p2.log" 2>&1 < /dev/null &

sleep 3
echo "started: eval daemon p1 (6 runs, 8 jobs) + p2 (24 runs, 16 jobs)"
