# Patches: getting `grasp_lift`/Allegro to build

`scripts/run_grasp_lift_probe.py` tries to use the paper's own task/env construction (Eden's
`grasp_lift` task, real grasp sampler) instead of a hand-rolled scene. As checked out, this repo's
task/env pipeline doesn't build for *any* task — these are the fixes needed to get it further,
roughly in increasing depth. See [docs/tactile-genesis-verification.md](../docs/tactile-genesis-verification.md)
for the full story, including where this stopped.

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

## 3. `src/registry.py` — `DEFAULT_RIGID_SOLVER_OPTIONS`

This pinned Genesis build (v1.4.1) takes `friction_cone`/`contact_resolution` nested under
`Scene(rigid_options=RigidOptions(...))`, not as flat `Scene(friction_cone=..., ...)` kwargs —
but `eden/envs/base.py` splats any `EnvOptions` extra field straight into `Scene(**scene_kwargs)`,
so a `RigidOptions` instance under the `"rigid_options"` key is what actually reaches `Scene()`
correctly. Also needs real enum instances (`gs.friction_cone.elliptic`, not the string
`"elliptic"`).

**Apply:** in `dexterous-hands/src/registry.py`, replace:
```python
DEFAULT_RIGID_SOLVER_OPTIONS = {
    "solver": "newton",
    "friction_cone": "elliptic",
    "contact_resolution": "signorini",
}
```
with:
```python
import genesis as _gs

DEFAULT_RIGID_SOLVER_OPTIONS = {
    "solver": "newton",
    "rigid_options": _gs.options.RigidOptions(
        friction_cone=_gs.friction_cone.elliptic,
        contact_resolution=_gs.contact_resolution.signorini,
    ),
}
```

## 4. `Eden/eden/envs/base.py` — deprecated `max_FPS` kwarg

`eden/envs/base.py` builds `ViewerOptions(max_FPS=int(1 / env_options.sim_dt), ...)`. This
Genesis build raises (not just warns) when `max_FPS` and `refresh_rate` are both considered set,
which happens here via Genesis's own internal option-propagation step. The deprecation warning
itself says `max_FPS` "now maps to `refresh_rate`", so using the new name directly is the same
value, not new behavior.

**Apply:** in `Eden/eden/envs/base.py`, change `max_FPS=int(1 / env_options.sim_dt)` to
`refresh_rate=int(1 / env_options.sim_dt)`.

## 5. Your own script needs `en.init(...)`

Not a repo bug — `main.py` calls `en.init(backend=..., log_root_path=...)` (which also handles
`gs.init()` internally) before building any config; a standalone script bypassing `main.py` needs
the same call first, or `gs.EPS`/Eden's logger are both unset and everything downstream fails.

## Where this stopped

One more layer in: `genesis/engine/solvers/base_solver.py` computes
`sim.dt / options.dt if "dt" in options.model_fields_set else sim.substeps`, and by this point
`options.dt` (on the `RigidOptions` instance from fix #3) is `None` while still showing up in
`model_fields_set` — a `model_copy_from`/`model_construct` interaction inside Genesis's own
pydantic Options framework, not something traceable to one obvious line. That's genuinely
framework-internal, stateful behavior rather than a renameable/reconstructable gap, so this is
where the reconstruction effort stopped.
