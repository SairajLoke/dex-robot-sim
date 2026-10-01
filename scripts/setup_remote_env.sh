#!/usr/bin/env bash
# Reproducible Tactile Genesis environment setup on a fresh GPU box (tested on a
# Vast.ai RTX 3060 Ti, CUDA 12.8, Python 3.12, uv). See ../docs/tactile-genesis-setup.md
# for the narrative version. Run from anywhere; everything lands under $WORKSPACE.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/tactile-genesis"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/neuroagents-lab/tactile-genesis.git "$REPO_DIR"
fi

cd "$REPO_DIR/dexterous-hands"

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# CUDA-matched torch wheel first; pyproject.toml leaves torch unpinned on purpose.
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# pt_tnn (PyTorchTNN, backing the optional tactile_convrnn encoder) isn't on any
# package index. The paper's students use --tactile_encoder=rnn, not convrnn, so it's
# never actually installed -- but `uv sync` still fully resolves the `convrnn` extra
# while building the lockfile, which fails without a source for it. Point uv at the
# upstream repo so resolution succeeds. Idempotent: skip if already patched.
if ! grep -q 'pt_tnn = ' pyproject.toml; then
  python3 - <<'EOF'
p = "pyproject.toml"
s = open(p).read()
anchor = "[tool.uv.sources]\n"
s = s.replace(anchor, anchor + 'pt_tnn = { git = "https://github.com/neuroagents-lab/PyTorchTNN" }\n', 1)
open(p, "w").write(s)
EOF
fi

uv sync
uv pip install pytest syrupy pytest-print -q

echo
echo "Setup complete. Activate with:"
echo "  source $REPO_DIR/dexterous-hands/.venv/bin/activate"
echo "Smoke test:"
echo "  cd $REPO_DIR/dexterous-hands && python main.py --task=in_hand_repose --robot=wuji --cpu --config=conf/experiments/tiny.yaml --no_loaded_video --no_checkpoint_video"
echo "Sensor physics test suite:"
echo "  cd $REPO_DIR/Genesis && pytest tests/sensors/test_tactile.py -v"
