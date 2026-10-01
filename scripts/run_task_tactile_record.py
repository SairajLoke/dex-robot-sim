"""Run one of the paper's own tasks (no policy training) and record tactile + auxiliary sensors.

in_fingers_rotate loads a sampled grasp on reset, so every fingertip starts in contact with the
object. Phase 1 holds with an all-zero action (hold the sampled grasp). Phase 2 (optional) adds a
small sinusoidal perturbation on the finger targets so the tactile field changes over time.

--squeeze_steps > 0 switches to a controlled press test instead of the wiggle: hold -> squeeze
(constant +squeeze_amp on every *bend* joint, accumulated into the target) -> hold -> release
(the opposite sign for as many steps), so deformation can be compared with contact force over a
wide, reversible range. --video also renders env 0 with an annotated overlay to <out>.mp4.

Output: <out>.npz written by the paper's TactileEpisodeRecorder (consumed by
scripts/animate_tactile.py), plus <out>_extra.npz with joint positions, object pose, privileged
contact / surface-distance probes, ground-truth hand<->object contact force per fingertip, rewards.

    python scripts/run_task_tactile_record.py --sensors actual-hand/force_torque --out rec_ft
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import genesis as gs
import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="in_fingers_rotate")
ap.add_argument("--robot", default="allegro")
ap.add_argument("--sensors", default="actual-hand/force_torque")
ap.add_argument("--hold_steps", type=int, default=100)
ap.add_argument("--wiggle_steps", type=int, default=200)
ap.add_argument("--wiggle_amp", type=float, default=0.3)
ap.add_argument("--squeeze_steps", type=int, default=0)
ap.add_argument("--squeeze_amp", type=float, default=0.1)
ap.add_argument("--settle_steps", type=int, default=40)
ap.add_argument("--video", action="store_true")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--out", default="task_tactile_record")
args = ap.parse_args()

en.init(backend=gs.gpu, log_root_path="logs/tactile_record")

from registry import get_task_config
from tactile_record import TactileEpisodeRecorder

config = get_task_config(
    run_name="tactile_record",
    task_name=args.task,
    modifiers={"robot": args.robot, "sensors": args.sensors},
    config_override_path="conf/experiments/tiny.yaml",
)
config.env_options._locked = False
config.env_options.num_envs = args.num_envs
if args.video:
    from eden.options.camera import CameraOptions, CamerasOptions

    config.cameras_options = config.cameras_options or CamerasOptions()
    config.cameras_options.rec = CameraOptions(
        cam_pos=(0.28, -0.32, 0.64), cam_lookat=(0.0, 0.0, 0.47), cam_fov=38.0, cam_resolution=(960, 720), debug=False
    )
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
u = env.unwrapped
obs, _ = env.reset()

robot, obj = u.entities["robot"], u.entities["obj"]
print("sensors:", sorted(u.sensors))
n_act = env.num_actions
print("num_actions:", n_act)

rec = TactileEpisodeRecorder(u, robot=args.robot, task=args.task)
print("recording:", rec.sensor_names)

priv = [n for n in sorted(u.sensors) if n.startswith("priv_")]
ex = {k: [] for k in ("joint_pos", "obj_pos", "obj_quat", "reward", "done")}
for n in priv:
    ex[n] = []


def _first(x):
    x = x[0] if isinstance(x, tuple) else x
    return torch.as_tensor(x)[0].float().reshape(-1).cpu().numpy()


t = 0.0
press = args.squeeze_steps > 0
total = args.hold_steps + (2 * args.squeeze_steps + args.settle_steps if press else args.wiggle_steps)
phase = np.random.RandomState(0).uniform(0, 2 * np.pi, n_act)
bend = torch.zeros(n_act, device=gs.device)
for i, n in enumerate([j.name for j in robot._entity.joints if j.n_dofs > 0]):
    if "bend" in n and i < n_act:
        bend[i] = 1.0
phase_log = []
writer = None
if args.video:
    import cv2

    cam = u.cameras["rec"]


def phase_name(step):
    if step < args.hold_steps:
        return "hold"
    if not press:
        return "wiggle"
    s = step - args.hold_steps
    if s < args.squeeze_steps:
        return "squeeze"
    if s < args.squeeze_steps + args.settle_steps:
        return "hold-squeezed"
    return "release"


for step in range(total):
    a = torch.zeros((args.num_envs, n_act), device=gs.device)
    pn = phase_name(step)
    if pn == "wiggle":
        s = step - args.hold_steps
        a += args.wiggle_amp * torch.as_tensor(np.sin(2 * np.pi * 0.5 * s * env.dt + phase), device=gs.device, dtype=a.dtype)
    elif pn == "squeeze":
        a += args.squeeze_amp * bend
    elif pn == "release":
        a -= args.squeeze_amp * bend
    phase_log.append(pn)
    obs, rew, done, _ = env.step(a)
    t += env.dt
    rec.record(t)
    ex["joint_pos"].append(robot.get_dofs_pos()[0].cpu().numpy())
    ex["obj_pos"].append(obj.get_pos()[0].cpu().numpy())
    ex["obj_quat"].append(obj.get_quat()[0].cpu().numpy())
    ex["reward"].append(float(rew[0]))
    ex["done"].append(bool(done[0]))
    for n in priv:
        ex[n].append(_first(u.sensors[n].read()))
    if args.video:
        rgb = cam.render_rgb()
        rgb = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
        frame = cv2.cvtColor(rgb.reshape(-1, *rgb.shape[-3:])[0][:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(f"{args.out}_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / env.dt, (frame.shape[1], frame.shape[0]))
        gt = {tip: float(np.linalg.norm(ex[f"priv_contact_{tip}_3_tip"][-1])) for tip in ("index", "middle", "ring", "thumb")}
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (30, 30, 30), thickness=-1)
        cv2.putText(frame, f"{args.sensors} | {pn} | t={t:5.2f}s | obj z={ex['obj_pos'][-1][2]:.3f}" + ("  [episode reset]" if done[0] else ""),
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "GT contact N  " + "  ".join(f"{k}={v:5.2f}" for k, v in gt.items()),
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 230, 255), 1, cv2.LINE_AA)
        writer.write(frame)
if writer is not None:
    writer.release()

path = rec.save(f"{args.out}.npz")
np.savez_compressed(f"{args.out}_extra.npz", t=np.asarray(rec.t, dtype=np.float32), phase=np.asarray(phase_log), **{k: np.asarray(v) for k, v in ex.items()})
print("saved", path, f"{args.out}_extra.npz")

d = np.load(path, allow_pickle=True)
for name in d["sensor_names"]:
    arr = d[f"{name}__data"]
    mag = np.linalg.norm(arr, axis=-1) if arr.ndim == 3 else np.abs(arr)
    print(f"{name}: shape={arr.shape} max={mag.max():.4f} hold_mean_max={mag[: args.hold_steps].max(axis=-1).mean():.4f}")
print("dones:", int(np.sum(ex["done"])), "obj z range:", np.min(np.asarray(ex["obj_pos"])[:, 2]), np.max(np.asarray(ex["obj_pos"])[:, 2]))
