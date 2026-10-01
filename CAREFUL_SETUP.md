# Careful setup (minimal)

Tactile Genesis (`neuroagents-lab/tactile-genesis`, Genesis v1.4.1) on a fresh Vast.ai GPU box (tested RTX 3060 Ti 8 GB, CUDA 12.8, Python 3.12, uv). `/workspace` is not persistent: redo this after a recycle.

## Steps

```bash
git clone git@github.com:SairajLoke/dex-robot-sim.git && cd dex-robot-sim
bash scripts/setup_remote_env.sh        # clone upstream, venv, torch cu128, uv sync, apply patches
source /workspace/tactile-genesis/dexterous-hands/.venv/bin/activate
cd /workspace/tactile-genesis/Genesis && pytest tests/sensors/test_tactile.py -v     # 24/24 expected
cd ../dexterous-hands                    # copy scripts/*.py here, then:
python run_task_tactile_record.py --sensors actual-hand/force_torque --out rec_ft     # or actual-hand/elastomer
python run_task_tactile_record.py --sensors actual-hand/elastomer --squeeze_steps 100 --num_envs 1 --video --out press_el
```

Always run on the GPU box, detached (`nohup ... &`), and call the venv's `python` explicitly.

## Bugs hit, and the fix (one pair per bug; full detail in `patches/README.md`)

| # | Bug (2 lines) | Fix (2 lines) |
|---|---|---|
| 1 | `uv sync` fails: `pt_tnn` (optional `convrnn` extra) is on no package index. | Point `[tool.uv.sources]` at the upstream PyTorchTNN git repo. Done by `setup_remote_env.sh`. |
| 2 | Every task config imports `eden.managers.terms.utils.soft_dof_pos_violation` and 3 quaternion helpers that are missing from the vendored Eden. | Reconstructed from their single call sites: new `terms/utils.py` plus `utils_geom.diff`. |
| 3 | `gs.Scene()` rejects `friction_cone` / `contact_resolution`; `ViewerOptions(max_FPS)` is deprecated (Genesis v1.4.1 moved them). | `envs_base.diff` routes them into Eden's own `RigidOptions` and renames `max_FPS` to `refresh_rate`. |
| 4 | Standalone scripts fail with an unset `gs.EPS` / logger. | Call `en.init(backend=gs.gpu, ...)` first, as `main.py` does. |
| 5 | Mass / COM shift methods and `termination_manager.get_term_dones` are missing in v1.4.1 / the snapshot. | `entities_rigid.diff` shims them with the absolute mass/COM setters; `termination_alias.diff` aliases `get_term_dones`. |
| 6 | `in_fingers_rotate` crashes: `ReferenceSource.ACCUMULATE` does not exist. | `accumulate_reference.diff`: persistent clamped target, reconstructed (not the authors' code). |
| 7 | `ApplyExternalForce/Torque` call removed Genesis APIs. | `external_wrench.diff` maps them to `apply_links_external_wrench`. |
| 8 | The 128-grasp cache file is not shipped and its loader silently skips, so the hand starts open and never touches the object. | `apply_patches.sh` symlinks it to the shipped `..._v141r2_grasps_32.pt`. |
| 9 | The authors' `animate_tactile.py` crashes on recordings: the recorder stores sensor history as `(T, 5*N, 3)` while positions are `(T, N, 3)`. | Median over the 5 history sub-steps first (`reduce_history` in our plot/animate scripts). |

## What was run (in order)

1. Setup + 24/24 sensor unit tests (closed-form checks pass).
2. Hand-built scene, one Allegro fingertip pressing an object: consistent force/torque.
3. Paper's `grasp_lift` env (after bugs 2-5): builds, but the scripted drive reads table contact, not object contact.
4. Paper's `in_fingers_rotate` (starts in contact, after bugs 6-8): records force_torque and elastomer, plus rendered video.

Results and caveats: `docs/tactile-genesis-verification.md`.
