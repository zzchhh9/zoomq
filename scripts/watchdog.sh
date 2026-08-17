#!/bin/bash
# Resume training arms that have died, from their own snapshot.pt.
#
#   setsid nohup ./scripts/watchdog.sh > logs/watchdog.log 2>&1 < /dev/null &
#
# WHY. `dishwasher_unload_cutlery` diverges in MuJoCo (`Nan, Inf or huge value
# in QACC at DOF 38` -> `mjWARN_BADQACC`) and takes the process down with it —
# six deaths so far, and ZERO on the other five tasks. The env truncates and
# resets on its own, but often enough the process is gone. Every arm writes
# `snapshot.pt` (save_snapshot=true), and the trainer resumes from it, so a
# dead arm loses only the frames since its last snapshot.
#
# THIS SCRIPT ONLY STARTS THINGS. It never kills, never pkills, never touches a
# process it did not start — a `pkill -f` whose pattern matches the ssh command
# line kills the caller's own shell, and none of these arms are safe to
# interrupt. Liveness is decided by `ps` on the run directory path.
#
# It reconstructs each arm's launch from `.hydra/hydra.yaml` (the overrides the
# arm was actually started with) rather than from a hardcoded table, so it stays
# correct as new waves are added.
set -uo pipefail

cd /mnt/workspace/zoomq
INTERVAL="${INTERVAL:-300}"
export DEMO_HOME=/mnt/workspace/zoomq/demos

while true; do
  for d in runs/p1_* runs/p2_* runs/p3_* runs/p4_*; do
    [ -d "$d" ] || continue
    a=$(basename "$d")
    # Alive? The run dir path appears in the trainer's own command line.
    if ps -eo args | grep -q "[/]runs/$a"; then continue; fi
    # Never resume something that never started (no snapshot to resume from).
    [ -f "$d/snapshot.pt" ] || { echo "$(date -Is) $a: no snapshot.pt, skipping"; continue; }

    hy="$d/.hydra/hydra.yaml"
    [ -f "$hy" ] || { echo "$(date -Is) $a: no hydra.yaml, skipping"; continue; }
    task=$(grep -oE "bigym_task=[a-z_0-9]+" "$hy" | head -1 | cut -d= -f2)
    seed=$(grep -oE "(^|[^a-z])seed=[0-9]+" "$hy" | head -1 | grep -oE "[0-9]+")
    [ -n "$task" ] || { echo "$(date -Is) $a: could not read bigym_task, skipping"; continue; }
    cfg="-"
    grep -q "config_zoomq_bigym" "$hy" && cfg="config_zoomq_bigym"
    # ZoomQ knobs and protocol flags this arm was launched with, replayed verbatim.
    extra=$(grep -oE "(zoomq\.[a-z_]+=[^ ,]+|temporal_ensemble=[a-z]+|nstep=[0-9]+)" "$hy" \
            | sort -u | tr "\n" " ")
    fr=$(tail -1 "$d/train.csv" 2>/dev/null | cut -d, -f1)

    echo "$(date -Is) RESUME $a task=$task seed=${seed:-1} cfg=$cfg extra='$extra' last_frame=$fr"
    env DEMO_HOME="$DEMO_HOME" SEED="${seed:-1}" EVAL_EVERY=10000 EVAL_EPS=25 WORKERS=8 \
      setsid nohup ./scripts/launch_arm.sh "$a" $((RANDOM % 10)) "$cfg" \
        bigym_task="$task" \
        lowerbody_policy.enabled=false \
        lowerbody_policy.base_action_mode=legacy_delta \
        lowerbody_policy.support_base_with_legs=false \
        success_hold_seconds=0 \
        in_train_eval=false save_eval_snapshot=true $extra \
        >> "logs/${a}.log" 2>&1 < /dev/null &
    sleep 5
  done
  sleep "$INTERVAL"
done
