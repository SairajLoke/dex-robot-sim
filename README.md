# dex-robot-sim

Dexterous robot simulation workspace.

## Tactile Genesis

Setup and verification of the [Tactile Genesis](https://neuroagents-lab.github.io/tactile-genesis/)
simulator ([arXiv:2606.22332](https://arxiv.org/pdf/2606.22332); anonymous ICRA submission at
`anonymous.4open.science/r/icra-tactile-genesis`, de-anonymized at
[`neuroagents-lab/tactile-genesis`](https://github.com/neuroagents-lab/tactile-genesis)):

- [CAREFUL_SETUP.md](CAREFUL_SETUP.md) — minimal setup steps and the 9 bugs hit, 2 lines each
- [docs/tactile-genesis-setup.md](docs/tactile-genesis-setup.md) — reproducible setup on a GPU box
  (also `scripts/setup_remote_env.sh`)
- [docs/tactile-genesis-verification.md](docs/tactile-genesis-verification.md) — verification that
  the torque sensing and tactile (elastomer) deformation implement real contact physics, not
  stubs: 24/24 of the repo's own sensor unit tests pass, plus a from-scratch demo pressing the
  paper's real Allegro hand asset onto a real object (`scripts/hand_object_tactile_demo.py`,
  `scripts/calibrate_allegro.py`, `scripts/calibrate_taxels.py`). Also documents 5 vendoring bugs
  found and fixed (plus a 6th where the fix chain stopped) that currently block the paper's own
  task/RL pipeline for every task, not just this one — see
  [patches/README.md](patches/README.md) for each fix and `scripts/run_grasp_lift_probe.py` for
  the probe script itself.
