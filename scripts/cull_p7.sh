#!/bin/bash
# Cull the 12 p7 mechanism arms. Approved by the user.
#
# They have delivered what they were for: 60K frames, 194-237 episodes, 0
# successes on every one, and the mechanism readouts are already recorded
# (delta_over_kappa_u p90 0.0099-0.1177 against the 1.0 the stopping rule needs;
# depth_share_0 0.918-0.920 with depths 1-4 sitting exactly on the annealed
# eps floor). Snapshots stay on disk, so any of them resumes.
#
# Kill by PID collected up front, never `pkill -f p7_...`: the pattern would
# appear in this session's own ssh command line and take the shell with it.
# Verified before running: the pattern matches 204 processes across exactly
# 12 names (zqS/zqSF/zqA x s1-s4) and nothing belonging to the other project.
set -u
PIDS=$(ps -eo pid,args --no-headers | grep -F "p7_mp_zq" | grep -v " grep " | awk '{print $1}')
n=$(echo "$PIDS" | grep -c . || true)
echo "collected $n pids"
[ "$n" -eq 0 ] && { echo "nothing to do"; exit 0; }
kill -TERM $PIDS 2>/dev/null || true
for i in $(seq 1 12); do
  sleep 5
  left=$(ps -eo pid,args --no-headers | grep -F "p7_mp_zq" | grep -v " grep " | wc -l)
  echo "  t+$((i*5))s: $left still up"
  [ "$left" -eq 0 ] && break
done
left=$(ps -eo pid,args --no-headers | grep -F "p7_mp_zq" | grep -v " grep " | awk '{print $1}')
if [ -n "$left" ]; then
  echo "SIGKILL for stragglers"
  kill -9 $left 2>/dev/null || true
  sleep 5
fi
echo "remaining: $(ps -eo pid,args --no-headers | grep -F "p7_mp_zq" | grep -v " grep " | wc -l)"
free -g | head -2
