#!/bin/bash
# Render the pixel demonstration cache for the Phase 1/2 tasks, SHARDED.
#
#   NSHARDS=12 ./scripts/build_cache.sh [--dry-run]
#
# BiGym demonstrations ship state-only; the pixel observations are rendered on
# first use inside one serial loop in `DemoStore.get_demos`. Measured on ali:
# 6.8-10.6 minutes per demo on the long tasks, i.e. 4-7 hours per task on a
# single core while ~150 cores sit idle. Demos are independent and `cache_demo`
# guards with `demo_exists` before writing each to its own path, so N sharded
# processes are safe and cut that to (4-7 h)/N.
#
# Do NOT "shard" by running N unsharded copies: `_get_demos` shuffles, so two
# processes can pass `demo_exists` for the same demo and both `write_bytes` to
# the same path while a third reads the half-written file.
#
# The env config here must match the training arms exactly, or the metadata
# differs and the cache is built under a path the runs will not look in.
set -euo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"
NSHARDS="${NSHARDS:-12}"

cd /mnt/workspace/zoomq
L=./scripts/launch_arm.sh
export DEMO_HOME=/mnt/workspace/zoomq/demos
FLOAT="lowerbody_policy.enabled=false lowerbody_policy.base_action_mode=legacy_delta lowerbody_policy.support_base_with_legs=false success_hold_seconds=0"

ppu=0
n=0
for spec in "mp:move_plate" "sh:saucepan_to_hob" "tc:take_cups" "pc:put_cups" \
            "du:dishwasher_unload_cutlery" "fc:flip_cutlery"; do
  short="${spec%%:*}"; task="${spec#*:}"
  for k in $(seq 0 $((NSHARDS - 1))); do
    $DRY env DEMO_HOME="$DEMO_HOME" EVAL_EVERY=999999 \
      setsid nohup $L "cache_${short}_${k}" "$ppu" - \
        bigym_task="$task" $FLOAT \
        demo_cache_only=true demo_cache_shard="$k" demo_cache_nshards="$NSHARDS" \
        > "logs/cache_${short}_${k}.log" 2>&1 < /dev/null &
    ppu=$(( (ppu + 1) % 16 ))
    n=$((n + 1))
    sleep 1
  done
done
echo "launched ${n} cache shards (${NSHARDS} per task x 6 tasks)"
