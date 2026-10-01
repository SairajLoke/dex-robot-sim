# Tactile Genesis — torque / tactile-deformation verification

Checks whether Tactile Genesis's torque sensing and tactile (elastomer) deformation actually
compute real contact physics, as claimed in the paper, rather than being stubbed or cosmetic.

**Setup:** see [tactile-genesis-setup.md](tactile-genesis-setup.md). Run on a Vast.ai RTX 3060 Ti
(8GB), commit `4777fd5d` of `neuroagents-lab/tactile-genesis`.

## What was run

The repo's own GPU sensor test suite, `Genesis/tests/sensors/test_tactile.py`:

```bash
cd Genesis
pytest tests/sensors/test_tactile.py -v --tb=short
```

**Result: 24 passed, 0 failed, 6 warnings, in 5m19s** (first run, including Taichi/Quadrants JIT
warm-up). No tests skipped, no xfails.

## What the paper claims, and what each test actually checks

The paper claims per-taxel kinematic force/torque, elastomer marker displacement (dilation +
shear), configurable sensor imperfections (crosstalk, hysteresis, dead taxels), and multiple
contact-query backends (SDF vs. raycast). The test suite exercises each of these with physical
assertions, not just "did it run":

**Kinematic force/torque** (`test_kinematic_contact_probe_box_sphere_support`) — a box
penetrates a ground plane by a known depth. The test asserts the reported force is *exactly*
`depth × stiffness × contact_normal` (closed-form match to the spring-damper model in
`kinematic_tactile.py`, not just "nonzero"), that un-contacted probes read exactly zero
force/torque, and that a `probe_gain` scales the measured (but not ground-truth) branch linearly.
Separately, `test_proximity_taxel_twist_torque` checks torque response to a twisting contact.

**Elastomer deformation** (`test_elastomer_sensor_sphere_ground_dilate_shear`,
`test_elastomer_sensor_grid_box_sphere`) — a sphere is pressed into a plane, then dragged
laterally. Assertions:
- Pure penetration → marker displacement points along the **outward surface normal** (dilation),
  zero shear component.
- Added lateral shift → shear component becomes tangential (zero projection onto the normal);
  dilation is unaffected.
- A sensor with both dilate and shear scales enabled reads the **sum** of the two isolated
  sensors to within `5e-5` — i.e. the combined elastomer signal is a true superposition of the
  two deformation modes, not an independent/approximate computation.
- Separating the sphere from the plane entirely drives the reading back to exactly zero.

**Sensor imperfections** (`test_kinematic_taxel_crosstalk`, `test_proximity_taxel_crosstalk`,
`test_contact_probe_hysteresis`, `test_contact_depth_probe_hysteresis_gain_and_dead_resample`) —
crosstalk kernels conserve total force (`rtol=5e-2` on the summed signal) while still blurring
individual taxel readings; the 9-neighbor crosstalk kernel leaks onto a probe that reads exactly
zero without it. Hysteresis state machine (latch-on/release thresholds) transitions correctly
through contact/no-contact cycles.

**Contact backends agree** (`test_contact_depth_query_sdf_vs_raycast_parity`) — SDF-based and
raycast-based contact queries on the same geometry report force directions with cosine similarity
`> 0.9`.

## Conclusion

On this hardware and commit, the torque and tactile-deformation machinery is not a stub: the
force/torque output matches the documented closed-form spring-damper model to numerical
tolerance, the elastomer dilation/shear channels decompose and superpose correctly under
controlled normal/lateral motion, and the configurable imperfection models (crosstalk,
hysteresis) behave as specified. This matches the sensing claims in the paper and on the project
site. Not verified here (out of scope for one 8GB consumer GPU): the throughput claims
(16,384 parallel envs, 600k steps/sec) and the FOTS/HydroShear marker-displacement RMSE
comparison, both of which require the full training/benchmark pipeline rather than the sensor
unit tests.

## Raw log

Full pytest output: `docs/tactile-genesis-test-tactile.log` (this directory).

## Follow-up: a real Allegro hand pressing a real object

The pytest suite above exercises the sensors against synthetic boxes/spheres directly, not
through the paper's actual dexterous-hand asset. To check the claims against a real robot, not
just the unit tests, I built `scripts/hand_object_tactile_demo.py`: load the paper's own Allegro
hand asset (`xela_v4/right_hand.xml`, 16 DOF) with its real 368-taxel fingertip layout
(`src/assets/sensors/allegro/actual/probes_368_hand_allegro.json`), press index/middle/ring
fingertips onto a fixed sphere via position-controlled joints, and read live `KinematicTaxel`
(force/torque) and `ElastomerTaxel` (deformation) output.

**This took far more iteration than the unit tests, for reasons worth recording:**

1. **Hand placement.** The `AllegroHand` metadata's `grasp_center` offset assumes Eden's own
   entity-placement step, which a raw `gs.morphs.MJCF(...)` load (used here to stay lightweight)
   doesn't apply — so a naive `hand_pos + grasp_center` object placement put the sphere nowhere
   near the open-pose fingertips. Fixed by an FK sweep (`scripts/calibrate_allegro.py`) of where
   fingertips actually land as bend joints close.

2. **Taxel surface vs. link origin.** The taxel probes live ~2cm forward/down of each fingertip
   link's own origin (the offset baked into `probe_local_pos`). Placing the object at the *link*
   position undershot the *taxel cloud* position by enough to miss contact entirely. Fixed with
   `scripts/calibrate_taxels.py`, which transforms each probe's local offset by the link's actual
   world quaternion to get the true taxel-surface centroid.

3. **Self-collision caps how far fingers actually close.** Under PD position control
   (`control_dofs_position`), the three bend joints per finger don't reach a shared target
   together — self-collision between curling finger segments (noted in Genesis's own startup
   warning: "Filtered out geometry pairs causing self-collision for the neutral configuration")
   stalls some joints well short of the commanded target while others overshoot. The achieved
   pose has to be measured empirically (by running and reading back `get_dofs_position`), not
   predicted from the control target.

4. **`KinematicTaxel` has no object filter, unlike `ElastomerTaxel`.** This was the key bug:
   `ElastomerTaxel` takes a `track_link_idx` restricting it to one target link, but
   `KinematicTaxel` has no equivalent — left unfiltered, it also reports contact from the hand's
   *own* self-collision (fingers/palm touching each other), not just the external object. This
   produced a force/torque reading on the middle finger that looked plausible but didn't
   correlate with actual distance to the sphere, while `ElastomerTaxel` (correctly restricted via
   `track_link_idx=(obj.base_link_idx,)`) stayed at zero the whole time — that mismatch is what
   exposed the bug. Fixed by passing `filter_link_idx` excluding every one of the hand's own
   links on each `KinematicTaxel` sensor, so only genuine external contact registers.

**Result after those fixes:** a free (non-fixed) object got knocked out of two fingers' reach by
real rigid-body contact dynamics once the third touched first, so the object was pinned
(`fixed=True`) to isolate the sensing question from grasp dynamics. The best run
(`docs/hand_object_demo_best_run.log`) shows the middle fingertip registering real, physically
consistent contact — force ≈1.16–1.43N with torque ≈0.07–0.10 (both scale together as expected
for an off-center contact patch, `torque ≈ r × F`), rising only once the finger closes onto the
sphere and reading exactly zero during the settle phase. Index/middle/ring simultaneous contact
with nonzero elastomer deformation on all three was not reached within the time available — each
geometry fix (self-collision-safe bend targets, taxel-surface placement) kept shifting where the
other two fingers needed the object to be, and iterating all three into alignment at once is a
multi-variable search I didn't finish closing.

## Follow-up: trying the paper's own grasp pipeline instead

Given the manual calibration above was fighting self-collision and placement issues that the
paper's own task/grasp-sampling code (`conf/sample_grasps/*.yaml`, the `grasp_lift` task) already
solves, I tried constructing their actual Eden task env directly
(`scripts/run_grasp_lift_probe.py`, via `registry.get_task_config` + `RslRlVecEnvWrapper`) instead
of continuing to hand-roll geometry. Note this never trains or runs a policy — `env.reset()` alone
is what gives a correctly-placed, paper-sampled grasp pose for free; stepping uses all-zero
actions purely to read sensor output over time.

**This repo's task/environment pipeline doesn't build as checked out, for any task** — not
specific to `grasp_lift`. Reconstructing just far enough to see that is itself the finding here.
Five issues, each one surfaced only after fixing the last, documented in full in
[patches/README.md](../patches/README.md):

1. `src/shared_terms.py` (imported by every task config) imports
   `eden.managers.terms.utils.soft_dof_pos_violation`, which doesn't exist anywhere in the
   vendored `Eden/` snapshot. Reconstructed from its one call site (a soft joint-limit penalty).
2. The same import line also needs 4 quaternion helpers from `eden.utils.geom`
   (`axis_angle_from_quat`, `inv_quat`, `quat_error_magnitude`, `quat_mul`) — none exist in the
   vendored copy. Reconstructed from the (w, x, y, z) convention already used elsewhere in that
   file and Genesis's own equivalent numpy implementations; sanity-checked standalone against
   known 90°/45° rotations before use.
3. A standalone script needs its own `en.init(...)` call before building any config (not a repo
   bug — `main.py` does this and I initially didn't).
4. This pinned Genesis build (v1.4.1) wants `friction_cone`/`contact_resolution` nested under
   `Scene(rigid_options=RigidOptions(...))`, not as flat kwargs the way this repo's own
   `src/registry.py` sets them.
5. `eden/envs/base.py` passes the deprecated `max_FPS=` kwarg to `ViewerOptions`, which this
   Genesis build hard-errors on (not just warns) once its own internal option-propagation also
   touches `refresh_rate`.

**Where it stopped:** one layer deeper, `genesis/engine/solvers/base_solver.py` computes
`sim.dt / options.dt if "dt" in options.model_fields_set else sim.substeps`. By this point the
`RigidOptions` instance from fix #4 has `dt=None` but is somehow *also* showing up in
`model_fields_set` — a `model_copy_from`/`model_construct` interaction inside Genesis's own
pydantic Options framework, not a renameable kwarg or a reconstructable missing function. That's
genuinely framework-internal, stateful behavior I can't safely patch without a much deeper dive
into how Genesis's Options classes propagate fields between each other, so this is where the
reconstruction effort stopped, per the standing instruction not to guess blindly at that depth.

Full tracebacks for each stage in `docs/grasp_lift_pipeline_bug.log` (original failure) — the
later stages aren't separately saved but are reproducible by applying `patches/` and re-running
`scripts/run_grasp_lift_probe.py`. This looks like an artifact of the vendoring/stripping process
described in the top-level README ("Eden... stripped to the modules this project imports")
combined with the vendored Eden snapshot predating some Genesis v1.4.1 API changes — worth
flagging upstream if you're in touch with the authors, since it blocks reproducing *any* of the
paper's actual tasks, not just this follow-up demo.

## Honest summary

- **Sensor physics (pytest suite): fully verified, high confidence.** 24/24 passed with
  closed-form and superposition-level assertions.
- **Real hand on a real object: partially verified.** One fingertip of the paper's actual Allegro
  asset shows correct, physically consistent force/torque from real contact with a real object,
  using the real 368-taxel layout. Full 3-finger simultaneous contact with nonzero deformation
  wasn't reached in the time available.
- **Paper's own task pipeline: 5 bugs found and fixed, stopped at a 6th.** None of the 5 fixed
  issues touch the sensing/physics claims themselves — they're all plumbing (missing functions,
  deprecated kwargs, option-construction order) between the vendored Eden and the vendored
  Genesis. The 6th is in Genesis's own Options/pydantic internals and would need real
  investigation, not a quick patch, to resolve safely. All 5 fixes are saved under `patches/` so
  a future attempt starts past this point rather than re-discovering it.
