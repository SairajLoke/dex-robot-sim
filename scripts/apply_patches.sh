#!/usr/bin/env bash
# Apply the Eden fixes in ../patches/eden to a fresh tactile-genesis checkout. Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/tactile-genesis}"
cd "$REPO_DIR"

TERMS="Eden/eden/managers/terms"
mkdir -p "$TERMS"
[ -f "$TERMS/utils.py" ] || cp "$HERE/patches/eden/managers_terms_utils.py" "$TERMS/utils.py"

for d in utils_geom envs_base entities_rigid termination_alias; do
  p="$HERE/patches/eden/$d.diff"
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $d"
  else
    git apply "$p" && echo "applied: $d"
  fi
done
