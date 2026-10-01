#!/usr/bin/env bash
# Apply the Eden fixes in ../patches/eden to a fresh tactile-genesis checkout. Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/tactile-genesis}"
cd "$REPO_DIR"

TERMS="Eden/eden/managers/terms"
mkdir -p "$TERMS"
[ -f "$TERMS/utils.py" ] || cp "$HERE/patches/eden/managers_terms_utils.py" "$TERMS/utils.py"

for d in utils_geom envs_base entities_rigid termination_alias accumulate_reference external_wrench; do
  p="$HERE/patches/eden/$d.diff"
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $d"
  else
    git apply "$p" && echo "applied: $d"
  fi
done

# in_fingers_rotate's LoadGraspPose event looks for a 128-grasp cache that is not shipped; it silently
# skips loading (hand starts open, no contact) if the file is missing. Point it at the 32-grasp cache
# that does ship (the v141r2 one is the one whose grasps are valid on Genesis v1.4.1).
GRASPS="dexterous-hands/src/assets/grasps"
if [ -f "$GRASPS/in_fingers_rotate_allegro_v141r2_grasps_32.pt" ] && [ ! -e "$GRASPS/in_fingers_rotate_allegro_mixed_grasps_128.pt" ]; then
  ln -s in_fingers_rotate_allegro_v141r2_grasps_32.pt "$GRASPS/in_fingers_rotate_allegro_mixed_grasps_128.pt"
  echo "linked grasp cache"
fi
