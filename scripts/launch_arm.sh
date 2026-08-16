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
#   EVAL_EPS (env, default 10)         — the repo default is 100; CQN-AS's
#     published protocol reports over 25 episodes.
#   SEED (env, default 1)              — for parallel multi-seed waves. Runs are
#     sequential in env steps (~1 frame/s ceiling), so parallelism cannot make
#     ONE run finish sooner; it can only make the seeds cost one run's wall time
#     instead of N. ali has 184 cores and a wave of ~2.5 cores per arm.
#   WORKERS (env, default 2)           — replay_buffer_num_workers. Keep it
#     matched across arms you intend to compare: it changes the sampling
#     interleaving, hence the RNG stream.
#   EVAL_EVERY (env, default 2500)     — an eval is 10 x 300 steps and never
#     terminates early on a task nobody can score on, so at the default cadence
#     it costs MORE wall clock than the training it interleaves. Set it high for
#     arms whose deliverable is the mechanism counters rather than a curve.
#   save_eval_snapshot=false           — 18 GB/run of eval snapshots buys
#     nothing tonight; snapshot.pt (resume) is still written.
set -euo pipefail

# CONFIG is a hydra --config-name (use "-" for the stock CQN-AS config). It has
# to be a separate argument because hydra's argparse rejects flags that appear
# after positional overrides.
ARM="${1:?usage: launch_arm.sh <ARM> <PPU> <CONFIG|-> [overrides...]}"
PPU="${2:?usage: launch_arm.sh <ARM> <PPU> <CONFIG|-> [overrides...]}"
CONFIG="${3:?usage: launch_arm.sh <ARM> <PPU> <CONFIG|-> [overrides...]}"
shift 3

CONFIG_ARGS=()
if [ "${CONFIG}" != "-" ]; then
  CONFIG_ARGS=(--config-name="${CONFIG}")
fi

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
  ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
  seed="${SEED:-1}" \
  save_snapshot=true save_eval_snapshot=false \
  num_eval_episodes="${EVAL_EPS:-10}" eval_every_frames="${EVAL_EVERY:-2500}" \
  save_video=false save_train_video=false \
  max_eval_success_videos=0 max_eval_failure_videos=0 \
  replay_buffer_num_workers="${WORKERS:-2}" device=cuda \
  experiment="zoomq_${ARM}" \
  hydra.run.dir="${RUNS}/${ARM}" \
  "$@"
