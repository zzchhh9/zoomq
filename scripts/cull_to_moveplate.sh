#!/bin/bash
# Reduce the fleet to move_plate only. USER-APPROVED.
#
#   ./scripts/cull_to_moveplate.sh --dry-run
#   ./scripts/cull_to_moveplate.sh --confirm
#
# WHY SPEED. `ps -eo pcpu` reports a LIFETIME AVERAGE, so the box looked like it
# was using 77 of 184 cores while arms ran at 0.21 fps. vmstat is the truth: run
# queue 271-356 against 184 cores, 80% user, 1% idle, no swap, 3 processes in D
# state. Two-times oversubscribed on CPU, caused by WORKERS=8 x 42 arms = 336
# dataloader workers. Halving the fleet roughly doubles fps; cutting it to 15
# should restore ~1.2 fps, i.e. 5-6x, and take the first mechanism datapoint from
# 2.5 hours to about 25 minutes.
#
# WHY move_plate, AND WHY THE REST CANNOT ANSWER ANYTHING YET. New eval points
# show the BASELINE on the primary band is also on the floor at 20K frames:
#     p2_tc_base_s1  10K 0.08 (2/25)  ->  21K 0.00 (0/25)
#     p2_tc_base_s2  10K 0.04         ->  20K 0.00
#     p2_pc_base_s1  10K 0.12 (3/25)  ->  20K 0.04 (1/25)
#     p2_pc_base_s2  10K 0.12         ->  20K 0.00
# The same baseline arm reads 0.08 then 0.00 at consecutive snapshots, so at a
# true rate near 5% those are two draws from one distribution. The published
# 14.0 / 22.5 are 100K-frame numbers and nothing on this box will reach 100K
# (~56 GB per arm at that point against 1800 GB total). So "ZoomQ 0.00 where the
# baseline is 0.04-0.20" was over-read from a single 10K snapshot: on take_cups,
# put_cups and flip_cutlery NEITHER method has left the floor, and those arms
# cannot discriminate anything at the +-6pp noise floor of a 25-episode eval.
#
# move_plate is the only task whose CQN-AS baseline is reproduced here
# (0.80 / 0.68 / 0.68 at 50K, mean 0.72, against the published 64.0 +- 7.5), so it
# is the only place a ZoomQ number is unambiguous. saucepan also reproduces
# (0.92 at 30K vs 80.5 +- 13.3) but has no ZoomQ arms, so it contributes nothing
# to the root-cause question and is stopped too.
#
# WHAT IS KEPT (15 arms, the whole live question):
#   p1_mp_s1/s2/s3      the reproduced baseline reference, already at 50K+
#   p3_mp_ctl_s1        CQN-AS under the matched execution protocol
#   p3_mp_zqA_s1/s2     ZoomQ as the paper describes it
#   p4_mp_zqE_s1/s2     ZoomQ under the stock protocol (completes the 2x2)
#   p5_mp_zqD_s1/s2     adaptive=false -- execute the full-depth chunk, i.e. the
#                       object the bootstrap already values
#   p5_mp_zqU_s1        uncertainty=target_gap -- no C51 atom-spacing floor
#   p6_mp_zqP_s1/s2     cond_mode=final -- the only config-reachable way to make
#                       Q^exec a function of the action
#   p6_mp_zqW_s2/s3     w_scale=1e6 -- reproduce arm C's dissociation (n=1 today)
#
# SAFETY. Whitelist, not blacklist: anything not named in KEEP is stopped, but the
# script prints the full list and requires --confirm. Matches `/runs/<arm>` from a
# script FILE, never a `pkill -f` pattern that could match the calling ssh command
# line (CLAUDE.md 2.7). Snapshots are left on disk, so any stopped arm can be
# resumed later from exactly where it was.
set -uo pipefail

KEEP=" p1_mp_s1 p1_mp_s2 p1_mp_s3 p3_mp_ctl_s1 p3_mp_zqA_s1 p3_mp_zqA_s2 p4_mp_zqE_s1 p4_mp_zqE_s2 p5_mp_zqD_s1 p5_mp_zqD_s2 p5_mp_zqU_s1 p6_mp_zqP_s1 p6_mp_zqP_s2 p6_mp_zqW_s2 p6_mp_zqW_s3 "

MODE="${1:---dry-run}"
[ "$MODE" = "--confirm" ] || MODE="--dry-run"
cd /mnt/workspace/zoomq

stopped=0
for d in runs/p[123456]_*; do
  [ -d "$d" ] || continue
  a=$(basename "$d")
  case "$KEEP" in *" $a "*) continue ;; esac
  pids=$(pgrep -f "/runs/${a}\b" || true)
  n=$(printf '%s\n' $pids | grep -c . || true)
  [ "$n" -eq 0 ] && continue
  fr=$(tail -1 "$d/train.csv" 2>/dev/null | cut -d, -f1 | cut -d. -f1)
  echo "[$MODE] stop $a  pids=$n  frame=${fr:-0}"
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
    pids=$(pgrep -f "/runs/${a}\b" || true)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  done
  echo "stopped $stopped arms; snapshots left on disk"
fi

echo "--- kept and alive:"
for a in $KEEP; do
  n=$(pgrep -cf "/runs/${a}\b" || true)
  printf "    %-16s pids=%s\n" "$a" "${n:-0}"
done
