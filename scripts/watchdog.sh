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
#
# MEMORY GATE. The first version of this script had two bugs that together made
# an OOM episode worse: it scanned only p1..p4, so the phase-5/6 arms were never
# covered; and it restarted arms unconditionally, including into a box that was
# already out of RAM. Measured PSS per arm is 17.9 GB below 5K frames and 33.0 GB
# above 40K, so a restart costs ~18 GB immediately and grows from there. It now
# refuses to restart anything unless MIN_FREE_GB is available, and it never
# restarts an arm on the CULL list.
set -uo pipefail

cd /mnt/workspace/zoomq
INTERVAL="${INTERVAL:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"
export DEMO_HOME=/mnt/workspace/zoomq/demos

# WHITELIST, not a blacklist. The previous version carried a CULLED blacklist and
# had to be edited after every cull; it was not, so it resurrected 27 arms that a
# later cull had deliberately stopped. The set of arms that SHOULD be running is
# small and stable, so name that instead: anything not listed here stays down, and
# a future cull needs no edit at all.
#
# These twelve are the CQN-AS baselines that reproduce the published numbers
# (move_plate 0.80/0.68/0.68 at 50K vs 64.0 +- 7.5; saucepan 0.92 at 30K vs
# 80.5 +- 13.3). Every ZoomQ arm is deliberately down: they all use the
# action-blind exec head, so more frames only re-confirm the same inert mechanism
# (see .claude/plans/cryptic-fluttering-treehouse.md). Their snapshots are intact,
# so any of them resumes by being added here.
KEEP=" p1_mp_s1 p1_mp_s2 p1_mp_s3 p1_sh_s1 p1_sh_s2 p1_sh_s3 p2_tc_base_s1 p2_tc_base_s2 p2_pc_base_s1 p2_pc_base_s2 p2_fc_base_s1 p2_fc_base_s2 "

while true; do
  for d in runs/p1_* runs/p2_* runs/p3_* runs/p4_* runs/p5_* runs/p6_*; do
    [ -d "$d" ] || continue
    a=$(basename "$d")
    case "$KEEP" in *" $a "*) ;; *) continue ;; esac
    # Alive? Match the TRAINER's own command line. A bare "/runs/$a" also matches
    # the eval daemons' `--path .../runs/$a` argument, which made dead arms look
    # alive during the OOM episode.
    if pgrep -f "train_cqn_as_bigym.*runs/${a}\b" >/dev/null 2>&1; then continue; fi
    free_gb=$(free -g | awk '/^Mem:/ {print $7}')
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
      echo "$(date -Is) $a is down but only ${free_gb} GB available (need ${MIN_FREE_GB}); NOT restarting"
      continue
    fi
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
