#!/bin/bash
# Automatic offline evaluation of the Phase 1/2 snapshots.
#
#   ./scripts/start_eval_daemons.sh
#
# The training arms run with `in_train_eval=false save_eval_snapshot=true`
# (CLAUDE.md's LAUNCH RULE), so they write a snapshot every 10k frames and pay
# no evaluation cost. This starts daemons that poll for new snapshots and score
# them on the otherwise-idle cores.
#
# --episodes 25 is CQN-AS's own reporting protocol ("success rate over 25
#   episodes"), which is what main.tex Table 1's numbers are.
# --videos 0 because a video eval has crashed a run here before with EGLError.
# --train-script is passed explicitly: these runs live in a dedicated results
#   tree, not exp_local/<method_dir>/, and the daemon otherwise skips them
#   forever with "unknown method dir" while looking perfectly healthy.
#
# SPREAD ACROSS CARDS, FEW JOBS EACH. A first attempt pinned 8 and 16 eval jobs
# to single PPUs and they died with `cudaGetDeviceCount ... out of memory` /
# `driver shutting down`: the cards were at only 25-28 GB of 98 GB, so this is
# context exhaustion, not memory. Training already places ~34 processes per card
# (each arm plus its 8 dataloader workers inherit CUDA_VISIBLE_DEVICES), so eval
# gets 4 jobs each on the least-loaded cards.
#
# The daemon spawns children with the inherited environment, so HOME (which is
# what bigym derives its demonstration cache from) and PYTHONUSERBASE (where
# mujoco lives, also derived from HOME) must both be exported here.
set -euo pipefail

REPO=/mnt/workspace/zoomq/third_party/CQN-AS-G1
RUNS=/mnt/workspace/zoomq/runs
LOGS=/mnt/workspace/zoomq/logs
JOBS="${JOBS:-4}"

set +eu
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
set -eu

export MUJOCO_GL=egl
# EGL enumerates a single device here, so pin index 0; reeval otherwise derives
# it from --gpus and mujoco refuses to start.
export MUJOCO_EGL_DEVICE_ID=0
export HOME=/mnt/workspace/zoomq/demos
export PYTHONUSERBASE=/root/.local
export EVAL_PYTHON=/mnt/workspace/anchorq/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

cd "$REPO"

# start_group <name> <gpu> <run...>
start_group() {
  local name="$1" gpu="$2"; shift 2
  local paths=()
  for a in "$@"; do paths+=(--path "$RUNS/$a"); done
  setsid nohup "$EVAL_PYTHON" scripts/eval_daemon.py \
    --gpu "$gpu" --jobs "$JOBS" --episodes 25 --videos 0 --oldest-first \
    --train-script train_cqn_as_bigym.py \
    "${paths[@]}" > "$LOGS/evald_${name}.log" 2>&1 < /dev/null &
  sleep 1
}

# Phase 1 is the gate, so it gets its own cards and is never queued behind the
# much longer Phase 2 episodes.
start_group p1a 15 p1_mp_s1 p1_mp_s2 p1_mp_s3
start_group p1b 14 p1_sh_s1 p1_sh_s2 p1_sh_s3

start_group p2a 13 p2_tc_base_s1 p2_tc_base_s2 p2_tc_zqA_s1 p2_tc_zqA_s2 p2_tc_zqF_s1 p2_tc_zqF_s2
start_group p2b 12 p2_pc_base_s1 p2_pc_base_s2 p2_pc_zqA_s1 p2_pc_zqA_s2 p2_pc_zqF_s1 p2_pc_zqF_s2
start_group p2c 11 p2_du_base_s1 p2_du_base_s2 p2_du_zqA_s1 p2_du_zqA_s2 p2_du_zqF_s1 p2_du_zqF_s2
start_group p2d 10 p2_fc_base_s1 p2_fc_base_s2 p2_fc_zqA_s1 p2_fc_zqA_s2 p2_fc_zqF_s1 p2_fc_zqF_s2

sleep 3
echo "started 6 eval daemons (${JOBS} jobs each) across PPUs 10-15"
