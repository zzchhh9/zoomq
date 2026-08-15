#!/bin/bash
# Launch one ZoomQ-programme training arm on ali.
#
#   ./launch_arm.sh <ARM> <PPU> [extra hydra overrides...]
#
# Deliberate choices, all of which the morning report has to be able to justify:
#   cwd is the SUBMODULE checkout, never /mnt/data/CQN-AS-G1 (same inode as
#     /mnt/workspace/CQN-AS-G1, the pi05-baseline tree with a live eval fleet).
#   save_video / save_train_video false  — a video eval once killed a run with
#     EGLError (eglDestroyContext); the repo default is true.
#   num_eval_episodes=10               — the repo default is 100.
#   save_eval_snapshot=false           — 18 GB/run of eval snapshots buys
#     nothing tonight; snapshot.pt (resume) is still written.
set -euo pipefail

ARM="${1:?usage: launch_arm.sh <ARM> <PPU> [overrides...]}"
PPU="${2:?usage: launch_arm.sh <ARM> <PPU> [overrides...]}"
shift 2

REPO=/mnt/workspace/zoomq/third_party/CQN-AS-G1
RUNS=/mnt/workspace/zoomq/runs
PY=/mnt/workspace/anchorq/.venv/bin/python

# The PPU SDK's envsetup.sh leaves a non-zero status and references unset vars,
# so errexit/nounset have to stand down for exactly this line. Without the
# sourcing, torch dies with `cannot open the file:libhggcrt1.so`.
set +eu
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
set -eu

export MUJOCO_GL=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${PPU}"

mkdir -p "${RUNS}"
cd "${REPO}"

exec "${PY}" train_cqn_as_bigym.py \
  seed=1 \
  save_snapshot=true save_eval_snapshot=false \
  num_eval_episodes=10 eval_every_frames=2500 \
  save_video=false save_train_video=false \
  max_eval_success_videos=0 max_eval_failure_videos=0 \
  replay_buffer_num_workers=2 device=cuda \
  experiment="zoomq_${ARM}" \
  hydra.run.dir="${RUNS}/${ARM}" \
  "$@"
