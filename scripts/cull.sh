#!/bin/bash
# Stop a named list of training arms, to free RAM. USER-APPROVED LIST ONLY.
#
#   ./scripts/cull.sh --dry-run     # show what would be stopped
#   ./scripts/cull.sh --confirm     # actually stop them
#
# WHY. The box is at 1720 of 1800 GB. Measured PSS per arm: 17.9 GB below 5K
# frames, 33.0 GB above 40K, i.e. roughly +23 GB more per arm to reach 100K.
# 47 arms x 23 GB = 1081 GB of growth against 71 GB free, so arms were dying to
# the OOM killer at random -- 13 so far, including six p2 arms with 40K+ frames
# of progress. Choosing which arms to stop is strictly better than letting the
# OOM killer choose.
#
# THE LIST (approved by the user 2026-08-17; each entry is an arm whose result
# would not have been reported anyway):
#   zqH x6  hold_dims. Refuted by open-loop demo replay before these arms were
#           launched: zero-order hold on the gripper dims gives +5pp on take_cups
#           and -5pp on put_cups at n=20, i.e. noise. See gates/g1v_*.json.
#   zqX x2  exec_lambda=0. Now known NOT to be the intended ablation: the exec
#           head is the sole bootstrap carrier and is zero-initialised, so
#           exec_lambda=0 pins it at zero forever and the critic degenerates into
#           a 16-step Monte-Carlo reward regressor (p4_tc_zqX_s1:
#           bellman_target_q == batch_reward == 0.0012367 to 5 s.f.).
#   du  x6  dishwasher_unload_cutlery. MuJoCo diverges there (`Nan, Inf or huge
#           value in QACC at DOF 38` -> mjWARN_BADQACC); six-plus process deaths
#           against ZERO on the other five tasks. The arms have been resumed from
#           snapshots repeatedly, so their training trajectories are not
#           comparable with the rest and their numbers could not be published.
#
# SAFETY. Never `pkill -f <pattern>` where the pattern could match the calling
# ssh command line (CLAUDE.md 2.7) -- that kills the caller's own shell. This
# script matches on the run-directory path, runs from a file rather than an ssh
# argument, requires --confirm, and refuses any name not on the list above.
set -uo pipefail

CULL=(
  p3_mp_zqH_s1 p3_mp_zqH_s2 p3_tc_zqH_s1 p3_tc_zqH_s2 p3_pc_zqH_s1 p3_pc_zqH_s2
  p4_tc_zqX_s1 p4_pc_zqX_s1
  p2_du_base_s1 p2_du_base_s2 p2_du_zqA_s1 p2_du_zqA_s2 p2_du_zqF_s1 p2_du_zqF_s2
)

MODE="${1:---dry-run}"
[ "$MODE" = "--confirm" ] || MODE="--dry-run"
cd /mnt/workspace/zoomq

total=0
for a in "${CULL[@]}"; do
  pids=$(pgrep -f "/runs/${a}\b" || true)
  n=$(printf '%s\n' $pids | grep -c . || true)
  fr=$(tail -1 "runs/$a/train.csv" 2>/dev/null | cut -d, -f1)
  echo "[$MODE] $a  pids=$n  last_frame=${fr:-none}"
  if [ "$MODE" = "--confirm" ] && [ -n "$pids" ]; then
    kill $pids 2>/dev/null
    total=$((total + n))
  fi
done

if [ "$MODE" = "--confirm" ]; then
  sleep 20
  # Anything that ignored SIGTERM.
  for a in "${CULL[@]}"; do
    pids=$(pgrep -f "/runs/${a}\b" || true)
    [ -n "$pids" ] && { echo "SIGKILL $a"; kill -9 $pids 2>/dev/null; }
  done
  echo "stopped ~$total processes across ${#CULL[@]} arms"
fi
