# Patches: getting `grasp_lift`/Allegro to build

`scripts/run_grasp_lift_probe.py` tries to use the paper's own task/env construction (Eden's
`grasp_lift` task, real grasp sampler) instead of a hand-rolled scene. As checked out, this repo's
task/env pipeline doesn't build for *any* task — these are the fixes needed to get it further,
roughly in increasing depth. See [docs/tactile-genesis-verification.md](../docs/tactile-genesis-verification.md)
for the full story, including what the readings show.

All of these are reconstructions/renames inferred directly from source (docstrings, single call
sites, deprecation messages) — not guesses at new behavior.

## 1. `eden/managers/terms/utils.py` — new file

`src/shared_terms.py` (imported by every task config) does
`from eden.managers.terms.utils import soft_dof_pos_violation`, and that module doesn't exist
anywhere in the vendored `Eden/` snapshot. Reconstructed from its one call site
(`shared_terms.dofs_pos_limits_penalty`): a soft joint-limit penalty, `0` inside a `[low, high]`
band and the clamped overshoot outside it.

**Apply:** copy [`eden/managers_terms_utils.py`](eden/managers_terms_utils.py) to
`Eden/eden/managers/terms/utils.py`.

## 2. `eden/utils/geom.py` — missing quaternion helpers

The same import line also needs `axis_angle_from_quat`, `inv_quat`, `quat_error_magnitude`,
`quat_mul`, none of which exist in the vendored `eden/utils/geom.py` (only `inv_quat` exists, in
*Genesis's* geom module, under a different package). These names match the IsaacLab/Orbit
math-utils convention Eden appears to follow.

**Apply:**
1. Add `inv_quat` to the existing `from genesis.utils.geom import (...)` block at the top of
   `Eden/eden/utils/geom.py`.
2. Append [`eden/utils_geom_additions.py`](eden/utils_geom_additions.py)'s three function
   definitions to the end of that same file.

## 3+4. `Eden/eden/envs/base.py` — `friction_cone`/`contact_resolution` routing and `max_FPS`

`registry.py` (unmodified) sets `friction_cone="elliptic"` and `contact_resolution="signorini"` as
extra `EnvOptions` fields, and `eden/envs/base.py` splats every extra straight into
`gs.Scene(**scene_kwargs)`. In this pinned Genesis (v1.4.1) those two are `RigidOptions` fields, not
`Scene` kwargs, so `Scene()` rejects them.

**Do not** work around this in `registry.py` by passing a `RigidOptions` under the extras key
`rigid_options`. That replaces the full `RigidOptions(dt=sim_dt, constraint_solver=..., ...)` Eden
builds, so `dt` and every other Eden solver setting is lost. That was an earlier version of this
patch, and it caused the `TypeError: float / NoneType` in `base_solver.py` that this directory used
to document as an unresolved Genesis bug. It was our own override, not a Genesis bug.

The fix is in `base.py`: pop those two extras, convert the strings to the `gs` enums, and pass them
into Eden's own `RigidOptions(...)` call. The same diff also renames the deprecated
`ViewerOptions(max_FPS=...)` to `refresh_rate=...` (the deprecation message says it maps to the
same thing; this Genesis hard-errors when both are considered set).

**Apply:** `cd Eden && git apply ../patches/eden/envs_base.diff`, or from the repo root of
tactile-genesis, `git apply` the diff as written. `registry.py` stays untouched.

## 5. Your own script needs `en.init(...)`

Not a repo bug — `main.py` calls `en.init(backend=..., log_root_path=...)` (which also handles
`gs.init()` internally) before building any config; a standalone script bypassing `main.py` needs
the same call first, or `gs.EPS`/Eden's logger are both unset and everything downstream fails.

## 6. `Eden/eden/entities/rigid.py` — `set_mass_shift` / `set_COM_shift` / `get_links_inertial_mass`

Eden delegates these to Genesis entity methods that v1.4.1 no longer has. Shims reimplement them with
the absolute `get_links_mass/set_links_mass` and `get_links_COM/set_links_COM`, caching nominal values
so shifts stay relative. Per-env values need `RigidOptions.batch_links_info=True`; with one env that is
not needed, and the shim raises for unbatched `num_envs > 1`.

## 7. `Eden/eden/managers/termination_manager.py` — `get_term_dones`

Called by the env but only `get_term` exists; added `get_term_dones = get_term`.

## 8. `ReferenceSource.ACCUMULATE` (`constants.py`, `actions/joint_actions.py`) — `accumulate_reference.diff`

`in_fingers_rotate` configures its `ExplicitPDController` with `reference_source=ACCUMULATE`, which
does not exist in the vendored Eden snapshot (`AttributeError`). Reconstructed from the authors' tests
and comments: a persistent per-env target, `target_t = clamp(target_{t-1} + action * scale)`, clamped
to the joint range minus a 5% margin each side, reset to the default pose on episode reset. This is a
reconstruction, not the authors' code; a zero action holds the pose, which is all the sensor recordings need.

## 9. External force / torque API (`events/domain_rand.py`) — `external_wrench.diff`

`ApplyExternalForce` / `ApplyExternalTorque` call `rigid_solver.apply_links_external_force/torque`,
which v1.4.1 replaced with `apply_links_external_wrench(force=, torque=, ...)`. The `ref` string is
mapped to `gs.link_ref_frame`.

## 10. Grasp cache symlink (`scripts/apply_patches.sh`)

`LoadGraspPose` skips loading silently if `in_fingers_rotate_allegro_mixed_grasps_128.pt` is missing, so
the hand would start open and nothing touches the object. The script symlinks it to the shipped
`..._v141r2_grasps_32.pt`.

## Applying

`scripts/apply_patches.sh` applies all of the above idempotently (`utils_geom.diff`, `envs_base.diff`,
`entities_rigid.diff`, `termination_alias.diff`, `accumulate_reference.diff`, `external_wrench.diff`, the grasp-cache symlink, plus copying `managers_terms_utils.py`). The manual
steps listed under fixes 1-2 are superseded by it.

## Where this stopped

The env builds, resets and steps. See `docs/tactile-genesis-verification.md` for what the tactile
readings do and do not show.
