#!/bin/bash
# Cull p8's four zqB/zqBS arms. Approved by the user.
#
# 37K frames, 125-129 episodes, 0 successes each. They test Stage B
# (refine_target=percell) -- but under temporal_ensemble=false nstep=16, the
# protocol that gives stock CQN-AS 0/160 on p8_mp_ctl_s1. A zero measured inside
# a regime that floors the baseline itself carries no information about Stage B,
# exactly as it carried none for p7. Snapshots stay on disk.
#
# p8_mp_ctl_s1/s2 STAY: they are the protocol control and the longest-running
# evidence on the box (0/160 and 17/160 at 172-176 episodes).
#
# Kill by PID collected up front, never `pkill -f p8_...`. Verified before
# firing: the pattern matches 68 processes across exactly four names and zero
# ctl processes.
set -u
PIDS=$(ps -eo pid,args --no-headers | grep -F "p8_mp_zqB" | grep -v " grep " | awk '{print $1}')
n=$(echo "$PIDS" | grep -c . || true)
echo "collected $n pids"
[ "$n" -eq 0 ] && { echo "nothing to do"; exit 0; }
kill -TERM $PIDS 2>/dev/null || true
for i in $(seq 1 12); do
  sleep 5
  left=$(ps -eo pid,args --no-headers | grep -F "p8_mp_zqB" | grep -v " grep " | wc -l)
  echo "  t+$((i*5))s: $left still up"
  [ "$left" -eq 0 ] && break
done
left=$(ps -eo pid,args --no-headers | grep -F "p8_mp_zqB" | grep -v " grep " | awk '{print $1}')
if [ -n "$left" ]; then kill -9 $left 2>/dev/null || true; sleep 5; fi
echo "zqB/zqBS remaining: $(ps -eo pid,args --no-headers | grep -F 'p8_mp_zqB' | grep -v ' grep ' | wc -l)"
echo "ctl still alive:    $(ps -eo pid,args --no-headers | grep -F 'p8_mp_ctl' | grep -v ' grep ' | wc -l) processes"
free -g | sed -n 2p
