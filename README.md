# dex-robot-sim

Dexterous robot simulation workspace.

## Tactile Genesis

Setup and verification of the [Tactile Genesis](https://neuroagents-lab.github.io/tactile-genesis/)
simulator ([arXiv:2606.22332](https://arxiv.org/pdf/2606.22332); anonymous ICRA submission at
`anonymous.4open.science/r/icra-tactile-genesis`, de-anonymized at
[`neuroagents-lab/tactile-genesis`](https://github.com/neuroagents-lab/tactile-genesis)):

- [docs/tactile-genesis-setup.md](docs/tactile-genesis-setup.md) — reproducible setup on a GPU box
  (also `scripts/setup_remote_env.sh`)
- [docs/tactile-genesis-verification.md](docs/tactile-genesis-verification.md) — verification that
  the torque sensing and tactile (elastomer) deformation implement real contact physics, not
  stubs: 24/24 of the repo's own sensor unit tests pass, plus a from-scratch demo pressing the
  paper's real Allegro hand asset onto a real object (`scripts/hand_object_tactile_demo.py`,
  `scripts/calibrate_allegro.py`, `scripts/calibrate_taxels.py`). Also documents a packaging bug
  found in the vendored repo that currently blocks the paper's own task/RL pipeline
  (`scripts/run_grasp_lift_probe.py`, `docs/grasp_lift_pipeline_bug.log`).
