"""Run one of the paper's own tasks (no policy training) and record tactile + auxiliary sensors.

in_fingers_rotate loads a sampled grasp on reset, so every fingertip starts in contact with the
object. Phase 1 holds with an all-zero action (hold the sampled grasp). Phase 2 (optional) adds a
small sinusoidal perturbation on the finger targets so the tactile field changes over time.

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
total = args.hold_steps + args.wiggle_steps
phase = np.random.RandomState(0).uniform(0, 2 * np.pi, n_act)
for step in range(total):
    a = torch.zeros((args.num_envs, n_act), device=gs.device)
    if step >= args.hold_steps:
        s = step - args.hold_steps
        a += args.wiggle_amp * torch.as_tensor(np.sin(2 * np.pi * 0.5 * s * env.dt + phase), device=gs.device, dtype=a.dtype)
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

path = rec.save(f"{args.out}.npz")
np.savez_compressed(f"{args.out}_extra.npz", t=np.asarray(rec.t, dtype=np.float32), **{k: np.asarray(v) for k, v in ex.items()})
print("saved", path, f"{args.out}_extra.npz")

d = np.load(path, allow_pickle=True)
for name in d["sensor_names"]:
    arr = d[f"{name}__data"]
    mag = np.linalg.norm(arr, axis=-1) if arr.ndim == 3 else np.abs(arr)
    print(f"{name}: shape={arr.shape} max={mag.max():.4f} hold_mean_max={mag[: args.hold_steps].max(axis=-1).mean():.4f}")
print("dones:", int(np.sum(ex["done"])), "obj z range:", np.min(np.asarray(ex["obj_pos"])[:, 2]), np.max(np.asarray(ex["obj_pos"])[:, 2]))
