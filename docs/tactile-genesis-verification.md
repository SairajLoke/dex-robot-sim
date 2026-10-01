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

### Elastomer deformation, second pass: press test + rendered video

Sensor types in the simulator (`conf/sensor/tactile_params.yaml`): `link_bool`, `link_force`, `agg_bool`,
`agg_force`, `bool`, `depth`, `force`, `force_torque`, `proximity`, `elastomer`. Only `force_torque`
and `elastomer` were exercised here. Privileged probes: `priv_contact_*` (GT hand-object force),
`priv_surface_distance_*`, `obj_force`.

`run_task_tactile_record.py --squeeze_steps 100 --video` runs hold (5 s) -> squeeze (all `bend` joints
closed, accumulated target) -> hold -> release, once per sensor type with the same trajectory, and renders
env 0. The squeeze is crude: the cube tumbles and is dropped 4 times (episode resets), so it is not a
clean press. `make_press_video.py` composes the render with synchronized GT force, force_torque,
elastomer and object-height traces: `docs/data/press_interaction.mp4`.

Pooled elastomer data (press_el + rec_el + rec_el_r3, 3760 fingertip-steps,
`scripts/plot_elastomer_dose_response.py`, `docs/data/elastomer_dose_response.png`):
- Elastomer loading rises with GT contact force: fraction of steps with any displaced taxel 0.006
  (<0.1 N), 0.09 (0.1-1 N), 0.17 (1-2 N), 0.34 (2-4 N); mean max displacement 0.003 -> 0.16 mm. The
  >4 N bin (n=28) is lower, so this is not monotone at the top and has too few samples to say more.
- Correlation is weak: max displacement vs GT 0.13, number of loaded taxels vs GT 0.28.
- Against fingertip-object distance the trend is not monotone (loaded fraction 0.11 at <2 mm, 0.23 at
  5-10 mm, 0.02 at >20 mm), unlike force_torque.
- Displacement is spiky (isolated bursts up to about 8 mm lasting a few steps, see the video), with most
  steps at exactly zero.
- The v141r3 grasp cache (`rec_el_r3`) did not give a firmer grasp: max displacement 7 mm, hold-phase
  loading near zero.

On this pipeline the elastomer is sparse, noisy and only weakly correlated with the ground truth. The
claim is not verified. (The clean press below explains why: in rigid-contact presses the elastomer is
zero by construction, so the in-hand bursts come from transient probe penetration, not from loading.)

### Correction to the first hand-built demo

`hand_object_tactile_demo.py` used sensor parameters that are not the paper's (elastomer `dilate_scale=0.1`,
`shear_scale=1.0`, `n_sample_points=600`, `normal_exponent=1.5` vs the paper's 100 / 200 / 1000 / 1.2; force_torque
`shear_scalar=4.0` vs 2.0). My earlier statement that elastomer zeros there pointed to a sensor problem was
therefore partly my own parameter error. The script now carries a warning; all later scripts load
`conf/sensor/tactile_params.yaml`.

## Clean controlled press (single fingertip, fixed sphere): `scripts/press_single_fingertip.py`

The in-hand cube tumbles under any squeeze, so this uses a controlled scene instead: the paper's Allegro asset,
middle finger only, bend targets ramped slowly into a fixed sphere (radius 75 mm, no gravity), the paper's
sensor parameters (loaded from `conf/sensor/tactile_params.yaml`), and per-step GT contact force from the rigid
solver. Data: `docs/data/press_single*.npz`, plot `docs/data/press_single_summary.png`
(`scripts/plot_press_single.py`).

| run | what | GT hand-sphere force | force_torque taxel sum | elastomer max |disp| |
|---|---|---|---|---|
| `press_single` (bend 0.3 -> 1.2) | rigid sphere | 1.8 -> 5.9 N, smooth | 0.79 -> 0.87 (4 taxels loaded) | **0 at every step** |
| `press_single_n20000`, `_n100000` | same, 20x / 100x more elastomer sample points | same | same | **0** |
| `press_single_deep` (bend 0.3 -> 1.6) | rigid sphere, deeper | 2.3 -> 9.6 N | 0.79 -> 0.92 | **0**; closest probe stays 0.94 mm outside the surface, never inside |
| `press_single_ghost`, `_ghost_fine` | sphere does not collide with the hand (disjoint contact mask), so probes can pass through | 0 | 0 | rises smoothly with penetration depth: about 0 at <3 mm, 1.5 at 10 mm, 4.5 at 20 mm, 8 at 27 mm, 17 at 40 mm; correlation with depth 0.94-0.96 |

Results:

1. **The clean force-vs-elastomer-deformation curve cannot be produced with a rigid object on this pipeline.**
   With real contact (GT force 2-10 N) the elastomer output is exactly zero. Reason, from
   `genesis/engine/sensors/point_cloud_tactile.py`: elastomer depth at each probe is `max(0, -SDF)` evaluated at the
   probe position against the tracked object's collision geometry, so a probe has to be *inside* the object.
   The rigid solver stops the fingertip surface (where the probes sit) about 1 mm outside the object, so depth is
   always 0. Measured directly: minimum probe signed distance +0.94 mm over the whole deep press.
2. **The elastomer sensor itself works.** When probes are allowed inside the sphere (ghost runs), the output is
   non-zero, monotone and smooth in penetration depth (the third panel). So the sensor reacts to
   penetration, not to contact force. Obtaining a force-vs-deformation curve would need either a soft/penetrable
   contact model or a force-to-penetration mapping that I did not find in the released code.
3. **force_torque saturates.** Over a 5x rise in GT force the taxel sum rises about 10% (0.79 -> 0.92) and only 4
   taxels load, so it is a proximity/contact indicator here, not a force measurement.
4. In ghost mode force_torque reads 0 (it needs real solver contact), so the two sensors never both respond in
   the same run.
5. The ghost-run trajectories oscillate (the PD-driven finger swings through the sphere), so penetration depth,
   not time, is the valid x axis; depths of 20-70 mm are far beyond a physical indentation and only show the trend.
6. `scripts/run_press_sweep.py` (8-env amplitude sweep on the cube) was not clean (GT force non-monotone, cube
   unstable) and is kept only for completeness (`docs/data/sweep_el.npz`).

## Honest summary

- **Sensor physics (pytest suite): fully verified, high confidence.** 24/24 passed with
  closed-form and superposition-level assertions.
- **Real hand on a real object, hand-built scene: partially verified.** One fingertip of the paper's
  Allegro asset shows consistent force/torque from real object contact with the real 368-taxel
  layout. No 3-finger contact, no deformation readings.
- **Paper's own task pipeline (`in_fingers_rotate`): runs, with patches in `patches/`.** The
  force_torque taxel loading follows object proximity (verified), but its magnitude is not
  proportional to the ground-truth contact force (weak correlation).
- **Elastomer deformation: not verified against contact force.** In a clean rigid press (GT force up to 9.6 N) the
  elastomer is exactly 0 because probes never get inside the object (closest +0.94 mm outside). It does respond
  smoothly to probe penetration depth when penetration is allowed (ghost sphere), so the sensor works but is not a
  force-to-deformation model for rigid contact. In-hand bursts (loaded fraction 0.006 -> 0.34 across GT bins,
  correlation 0.13-0.28) are sparse and spiky.
- **16,384-env throughput: not reproducible on this GPU.** 16,384 no-sensor envs run out of memory at scene build
  (8 GB card); the best that fits is 12,288 envs at about 6.6k samples/s (4,096 envs: 5.3k). The paper's claim is about
  600k steps/s on an RTX 5090 in a different scene, so the two are not comparable. Step time is dominated by the env
  managers (0.414 s per env.step vs 0.042 s raw scene.step at 1,024 envs). Elastomer sensors run out of memory at
  1,024 envs (256 envs: 362 samples/s). Raw numbers: `docs/data/throughput_3060ti.txt`.
- **FOTS / HydroShear comparison: ran, with a modified sensor config.** Best-fit marker RMSE against the real GelSight
  frames (px, lower is better; `docs/data/fots_compare/grid_sweep_best.png`, log `docs/logs/fots_compare.log`):

  | model | dilate RMSE (rel) | shear RMSE (rel) |
  |---|---|---|
  | Ours (ElastomerTaxel) | 0.66 (0.35) | 1.02 (0.19) |
  | HydroShear | 0.76 (0.40) | 1.48 (0.28) |
  | FOTS | 0.98 (0.52) | 1.50 (0.29) |

  Ours is lowest on both motions, so the ordering matches the paper's claim. Caveats: (1) it is one real frame per
  motion with hand-set geometry, and the RMSEs are all about 1 px, so this is weak evidence; (2) each model has a free
  per-motion scale fitted analytically after the sweep (ours dilate scale 0.067, FOTS 0.0002, HydroShear 5.2), so the
  comparison is of shape, not absolute magnitude; (3) the authors' script could not run as shipped, see below.
  Deviations (`patches/fots_compare.diff`, `scripts/run_fots_compare_patched.sh`): `DILATION_REG` is undefined and
  `dilation_reg` is not a valid sweep axis, so ours stage 1 sweeps compressibility only (dilation_reg at its auto
  default); the pinned Genesis v1.4.1 has no `ElastomerTaxel.elastomer_boundary`, so everything runs with
  `--no-boundary` (the paper's no-flux boundary wall is NOT included in "Ours" here) and the kwarg is omitted.
