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
