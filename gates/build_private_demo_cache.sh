#!/bin/bash
# Mirror the read-only BiGym demo cache into a PRIVATE tree under gates/, with
# every .safetensors as a symlink.  Directories are real, so any new
# "<freq>hz" cache that DemoStore generates lands here and never touches
# /mnt/data/bigym-cache.
set -euo pipefail
SRC=/mnt/data/bigym-cache/demonstrations/0.9.0
DST=/mnt/workspace/zoomq/gates/bigym_cache/demonstrations/0.9.0
mkdir -p "$DST"
touch "$DST/.lock"          # DemoStore.cached -> no wget download attempt
for TASK in MovePlate DrawerTopClose; do
  cd "$SRC"
  find "$TASK" -type d -print0 | while IFS= read -r -d '' d; do
    mkdir -p "$DST/$d"
  done
  find "$TASK" -type f -name '*.safetensors' -print0 | while IFS= read -r -d '' f; do
    ln -sf "$SRC/$f" "$DST/$f"
  done
done
echo "--- private cache ---"
find "$DST" -type d | sort
for d in $(find "$DST" -type d); do
  n=$(ls "$d"/*.safetensors 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "$n  ${d#$DST/}"
done
echo "--- source untouched (mtime) ---"
stat -c '%y %n' "$SRC" "$SRC/MovePlate" "$SRC/DrawerTopClose"
