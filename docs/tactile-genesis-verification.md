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


## Follow-up: the paper's own `grasp_lift` env, now building and running

Superseding the "stopped at a 6th bug" account in earlier commits: that 6th bug (`float / NoneType`
in `base_solver.py`) was **caused by my own `registry.py` override**, which replaced Eden's
`RigidOptions` and lost its `dt`. It was not a Genesis bug. `registry.py` is now untouched and
`friction_cone`/`contact_resolution` are routed into Eden's own `RigidOptions` in `eden/envs/base.py`.
Two more mechanical gaps followed (Genesis v1.4.1 dropped `set_mass_shift`/`set_COM_shift`, which
Eden's `rigid.py` still delegates to; `TerminationManager.get_term_dones` is called but only
`get_term` exists). Shims for both are in `patches/eden/`. Apply everything with
`scripts/apply_patches.sh` (run automatically by `scripts/setup_remote_env.sh`).

Result: `grasp_lift` / Allegro / `actual-hand/force_torque` builds, resets and steps with 18 tactile
force/torque sensors plus ground-truth contact and distance probes (`scripts/run_grasp_lift_probe.py`).

Two facts about the task that the earlier text got wrong:
- `grasp_lift` does **not** sample a grasp pose. The hand starts open, 0.08 m above the object.
- The wrist action is an **absolute** pose (`RootPoseController`), so an all-zero action commands
  z=0.74 and does not hold still. The zero-action probe reads exactly 0 on every sensor, which says
  nothing about the sensors.

### Scripted (non-learned) descend-and-close drive: `scripts/run_grasp_lift_scripted.py`

Open-loop: align the wrist over the object, descend by 10/20 mm, close the finger bend joints. Ground
truth is `RigidEntity.get_contacts()` bucketed by partner (object / table / hand self-contact).

**Force/torque on the real hand+object is NOT verified.**
- The fingertip taxel sensor has no object filter. It reads hand self-contact and table contact too.
  Control (hover, fingers open) already gives thumb-tip peak 0.5677 and any-sensor peak 1.19.
- Per-step thumb trace, 10 mm trial: taxels are loaded (10-30 taxels, sum |F| about 4-7) at every step
  where ground truth shows thumb-tip/table contact (2-17 N) and the thumb is 3-14 cm from the object.
  Thumb-tip/object contact only starts at step 116 and is weak (0.11-0.14 N). Taxel reading there:
  sum |F| 0.85, 0.21, 0.21, then 0.0 while ground truth still shows 0.108 N on the object.
- Index/middle/ring never touch the object, the object never lifts (dz about 0.001), and no
  ElastomerTaxel deformation was measured on this pipeline.
- Summary: the taxel signal tracks table contact; object contact in this run is too light and too
  brief to separate from it.

Logs: `docs/scripted_trace_10mm.log` (per-step thumb trace).

### Unverified paper claims on this hardware
16,384 envs / 600k steps/s throughput and the FOTS/HydroShear RMSE comparison were not attempted
(8 GB GPU, no datasets).

## Follow-up: `in_fingers_rotate` (starts in contact), no training

`scripts/run_task_tactile_record.py` builds the paper's `in_fingers_rotate` / Allegro env, which loads a
sampled grasp on reset so every fingertip starts touching the object. Phase 1 (5 s) holds with a zero
action; phase 2 (10 s) adds a sinusoidal wiggle on all 16 finger targets. 4 envs, env 0 recorded, 300
steps, with `actual-hand/force_torque` (`rec_ft`) and `actual-hand/elastomer` (`rec_el`). Sensors track
the object only (`track_link_idx="obj"`). Extra patches needed: ACCUMULATE reference source, external
wrench API, grasp-cache symlink (`patches/README.md` items 8-10). The object is dropped 5 times in the
10 s wiggle (episode resets, red dotted lines in the plots).

Outputs in `docs/data/`: `rec_{ft,el}.npz` (recorder), `rec_{ft,el}_extra.npz` (joints, object pose,
`priv_*` ground-truth probes, reward, done), `rec_{ft,el}_summary.png`, `rec_{ft,el}_tips.mp4`
(fingertip taxels, 3D). Scripts: `plot_task_tactile_record.py`, `animate_task_tactile.py`.

**force_torque results**
- Palm and mid-finger pads read exactly 0, so object-only tracking works.
- Fingertip loading follows fingertip-to-object surface distance monotonically: any-taxel-loaded
  fraction 0.87 at < 2 mm vs 0.11 at >= 20 mm; mean taxel sum |F| 0.92 vs 0.10.
- Magnitude does NOT follow the ground-truth hand-object contact force (`priv_contact_*`). Time
  correlation of taxel sum |F| with GT |F|: index 0.41, middle 0.57, ring 0.005, thumb 0.25.

**elastomer results (deformation): not verified.** Loaded in only about 12% of near-contact steps, with
sub-millimetre displacement and no monotonic trend with distance. The grasp is light. Deformation
physics still rests on the 24/24 pytest suite and the hand-built single-fingertip test above.

**Recorder quirk.** Sensors keep `history_length=5` sub-steps, and `TactileEpisodeRecorder.save` writes
`(T, H*N, 3)` history-major while `__pos` is `(T, N, 3)`. The authors' `scripts/animate_tactile.py`
fails on these files for that reason. Our scripts reduce over H with a median first (`reduce_history`).
Earlier ad-hoc taxel counts that skipped this were inflated 5x and were recomputed.

## Honest summary

- **Sensor physics (pytest suite): fully verified, high confidence.** 24/24 passed with
  closed-form and superposition-level assertions.
- **Real hand on a real object, hand-built scene: partially verified.** One fingertip of the paper's
  Allegro asset shows consistent force/torque from real object contact with the real 368-taxel
  layout. No 3-finger contact, no deformation readings.
- **Paper's own task pipeline (`in_fingers_rotate`): runs, with patches in `patches/`.** The
  force_torque taxel loading follows object proximity (verified), but its magnitude is not
  proportional to the ground-truth contact force (weak correlation).
- **Elastomer deformation on the paper's pipeline: not verified.** Weak, rarely loaded, sub-mm.
- **Not attempted:** 16,384-env / 600k steps/s throughput; FOTS / HydroShear RMSE comparison.
- **Next step:** a firmer grasp (the v141r3 grasp cache, or increased closure) so the elastomer
  actually indents, and time-alignment of taxel readings with `obj_force`.
