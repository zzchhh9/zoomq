#!/bin/bash
# Stop every ZoomQ arm; keep only the CQN-AS baselines. USER-APPROVED.
#
#   ./scripts/cull_to_baselines.sh --dry-run
#   ./scripts/cull_to_baselines.sh --confirm
#
# WHY. `zoomq.tex` §5: "Nothing below the line trains until Gates 0-2 pass."
# Gate 1 has passed (gates/g1v_*.json: a j=3 skeleton reaches the j=16 ceiling on
# take_cups and put_cups). Gate 2 cannot be run yet, because it reads Q^exec and
# Q^exec is not a function of the action: measured on a trained snapshot,
# dQ^exec_0/d(action) = 0.000000e+00 exactly, and Q^exec moves only 0.064 C51
# atoms across six wildly different chunks while the same network's per-cell head
# moves 3.039. Every arm below the line uses that head, so more frames only
# re-confirm the same inert mechanism.
#
# KEPT (12) -- the baselines that reproduce the published CQN-AS numbers and are
# the required control for anything measured later:
#   p1_mp_s1/s2/s3     move_plate    0.80 / 0.68 / 0.68 at 50K   (published 64.0 +- 7.5)
#   p1_sh_s1/s2/s3     saucepan      0.92 at 30K                 (published 80.5 +- 13.3)
#   p2_{tc,pc,fc}_base_s1/s2         the per-task references
#
# STOPPED -- every zqA / zqF / zqE / zqD / zqU / zqW arm, and the `ctl` arms.
# `ctl` is stock CQN-AS under the matched execution protocol; it is not a ZoomQ
# arm, but the deployment half of that question is already answered (frozen-weight
# paired eval, 0.68 vs 0.64 at n=25 -- a null), and the training half is not on the
# critical path for the exec-head fix. Its snapshots are kept, so it resumes by
# name whenever it is wanted.
#
# NOTHING IS DELETED. Every stopped arm keeps its snapshot.pt, so all of this is
# reversible: add the name to scripts/watchdog.sh's KEEP list and it comes back
# from where it left off.
#
# SAFETY. Whitelist, not blacklist. Prints the full list and requires --confirm.
# Matches the TRAINER's command line from a script FILE -- never a `pkill -f`
# pattern that could match the calling ssh command line (CLAUDE.md 2.7), and never
# a bare `/runs/<arm>` which also matches the eval daemons' --path argument.
set -uo pipefail

KEEP=" p1_mp_s1 p1_mp_s2 p1_mp_s3 p1_sh_s1 p1_sh_s2 p1_sh_s3 p2_tc_base_s1 p2_tc_base_s2 p2_pc_base_s1 p2_pc_base_s2 p2_fc_base_s1 p2_fc_base_s2 "

MODE="${1:---dry-run}"
[ "$MODE" = "--confirm" ] || MODE="--dry-run"
cd /mnt/workspace/zoomq

stopped=0
for d in runs/p[123456]_*; do
  [ -d "$d" ] || continue
  a=$(basename "$d")
  case "$KEEP" in *" $a "*) continue ;; esac
  pids=$(pgrep -f "train_cqn_as_bigym.*runs/${a}\b" || true)
  [ -z "$pids" ] && continue
  n=$(printf '%s\n' $pids | grep -c .)
  fr=$(tail -1 "$d/train.csv" 2>/dev/null | cut -d, -f1 | cut -d. -f1)
  snap=$([ -f "$d/snapshot.pt" ] && echo yes || echo NO)
  echo "[$MODE] stop $a  pids=$n  frame=${fr:-0}  resumable=$snap"
  if [ "$MODE" = "--confirm" ]; then
    kill $pids 2>/dev/null
    stopped=$((stopped + 1))
  fi
done

if [ "$MODE" = "--confirm" ]; then
  sleep 25
  for d in runs/p[123456]_*; do
    a=$(basename "$d")
    case "$KEEP" in *" $a "*) continue ;; esac
    pids=$(pgrep -f "train_cqn_as_bigym.*runs/${a}\b" || true)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  done
  echo "stopped $stopped arms; every snapshot left on disk"
fi

echo "--- kept:"
for a in $KEEP; do
  n=$(pgrep -cf "train_cqn_as_bigym.*runs/${a}\b" 2>/dev/null || echo 0)
  printf "    %-16s trainer_pids=%s\n" "$a" "${n:-0}"
done
