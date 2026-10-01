# Tactile Genesis — remote GPU setup

Reproducible setup for the [Tactile Genesis](https://neuroagents-lab.github.io/tactile-genesis/)
paper's simulator (ICRA submission, [arXiv:2606.22332](https://arxiv.org/pdf/2606.22332)) on a
Vast.ai GPU instance. The double-blind anonymous snapshot
(`https://anonymous.4open.science/r/icra-tactile-genesis`) and the de-anonymized public source
(`https://github.com/neuroagents-lab/tactile-genesis`) are the same monorepo; this guide clones
the public one since it's directly `git clone`-able.

## What this is

A self-contained monorepo that vendors every dependency its results were measured against:

| Directory | Package | Role |
|---|---|---|
| `dexterous-hands/` | `tactile_genesis` | Tasks, hands, sensors, training entrypoints |
| `Eden/` | `eden` | RL environment framework on Genesis + rsl_rl |
| `Genesis/` | `genesis-world` | Physics backend — **every tactile sensor lives here** |
| `rsl_rl/` | `rsl-rl-lib` | PPO + teacher/student distillation loop |

Pinned at commit `4777fd5d` (2026-09-18), Genesis v1.4.1.

## Environment used

- Vast.ai instance, RTX 3060 Ti (8GB VRAM), CUDA 12.8, Python 3.12.14, `uv` package manager.
- `/workspace` on this instance is **not** a persistent volume — everything here is lost on
  recycle/destroy. Re-run this setup (or sync `/workspace` off-box) if the instance is recycled.

## Steps

```bash
# 1. Clone
cd /workspace
git clone https://github.com/neuroagents-lab/tactile-genesis.git
cd tactile-genesis/dexterous-hands

# 2. Install the CUDA-matched torch wheel first (pyproject.toml leaves torch unpinned
#    on purpose, since the right wheel depends on the host's driver/CUDA version)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. One local patch needed: `pt_tnn` (PyTorchTNN, backing the optional tactile_convrnn
#    encoder) isn't published on any package index. The paper's students all use
#    --tactile_encoder=rnn, not convrnn, so it's never actually installed — but `uv sync`
#    still fully resolves the `convrnn` extra while building the lockfile, which fails
#    without a source for it. Point uv at the upstream repo so resolution succeeds:
cat >> pyproject.toml <<'EOF'
pt_tnn = { git = "https://github.com/neuroagents-lab/PyTorchTNN" }
EOF
# (insert the line under the existing [tool.uv.sources] table, not as a new table)

uv sync
```

## Smoke test

```bash
cd dexterous-hands
python main.py --task=in_hand_repose --robot=wuji --cpu \
  --config=conf/experiments/tiny.yaml --no_loaded_video --no_checkpoint_video
```

## Remote access pattern used here

```bash
ssh -p <port> root@<host> -L 8080:localhost:8080
```
then drive setup/tests non-interactively over that same SSH connection (`ssh ... "command"`),
since the instance's `/workspace` is ephemeral and the GPU work doesn't need a local display.
