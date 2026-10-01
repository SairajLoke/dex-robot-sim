"""
Use the paper's OWN task/env construction (Eden + the grasp_lift task's real grasp
sampler for Allegro) instead of a hand-rolled scene, to read real tactile sensor
output from a paper-faithful episode. Reuses their TactileEpisodeRecorder (the same
code `main.py --mode=play --save_tactile` uses) rather than reading sensors manually.
"""

import sys

import torch

sys.path.insert(0, "src")
import genesis as gs
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

from registry import get_task_config
from tactile_record import TactileEpisodeRecorder

# get_argparser()/get_task_config_from_args() eagerly load EVERY registered task's
# modifiers to build the full CLI parser, including in_fingers_rotate, whose config.py
# imports eden.managers.terms.utils.soft_dof_pos_violation -- a function that does not
# exist anywhere in this vendored Eden snapshot, so that import always fails. Calling
# get_task_config() directly only touches grasp_lift's own config module.
config = get_task_config(
    run_name="tactile_probe",
    task_name="grasp_lift",
    modifiers={"robot": "allegro", "stage": 1, "sensors": "actual-hand/force_torque"},
    config_override_path="conf/experiments/tiny.yaml",
)
config.env_options._locked = False
config.env_options.num_envs = 1

env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
obs, _ = env.reset()

target = env.unwrapped if hasattr(env, "unwrapped") else env
print("ENV sensors:", sorted(getattr(target, "sensors", {}).keys()))

recorder = TactileEpisodeRecorder(target, robot="allegro", task="grasp_lift")
print("RECORDER sensor_names:", recorder.sensor_names)

n_actions = getattr(env, "num_actions", None)
if n_actions is None:
    n_actions = env.action_space.shape[-1]
print("num_actions:", n_actions)
actions = torch.zeros((1, n_actions), device=gs.device)

t = 0.0
for i in range(150):
    obs, *_ = env.step(actions)
    t += env.dt
    recorder.record(t)

path = recorder.save("grasp_lift_tactile_probe.npz")
print("Saved:", path)

import numpy as np

d = np.load(path, allow_pickle=True)
print("\nnpz keys:", list(d.keys())[:20])
for name in d["sensor_names"]:
    arr = d[f"{name}__data"]
    mag = np.linalg.norm(arr, axis=-1) if arr.ndim == 3 else np.abs(arr)
    print(f"{name}: shape={arr.shape} max={mag.max():.5f} max_at_t0={mag[0].max():.5f} max_at_tN={mag[-1].max():.5f}")
