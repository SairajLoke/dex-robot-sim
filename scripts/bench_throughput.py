"""Env-step throughput of the paper's in_fingers_rotate/Allegro env at one num_envs (run once per N, separate process).

    python scripts/bench_throughput.py --num_envs 1024 --sensors actual-hand/force_torque
Prints one JSON line: env steps/s, env-steps*num_envs/s, peak GPU memory. Zero actions, no rendering, no learning.
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "src")
import genesis as gs
import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="in_fingers_rotate")
ap.add_argument("--robot", default="allegro")
ap.add_argument("--sensors", default="none")
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--steps", type=int, default=100)
args = ap.parse_args()

en.init(backend=gs.gpu, log_root_path="logs/bench")
from registry import get_task_config

mods = {"robot": args.robot}
if args.sensors != "none":
    mods["sensors"] = args.sensors
config = get_task_config(run_name="bench", task_name=args.task, modifiers=mods, config_override_path="conf/experiments/tiny.yaml")
config.env_options._locked = False
config.env_options.num_envs = args.num_envs
t0 = time.time()
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
obs, _ = env.reset()
torch.cuda.synchronize()
build_s = time.time() - t0
a = torch.zeros((args.num_envs, env.num_actions), device=gs.device)
for _ in range(args.warmup):
    env.step(a)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(args.steps):
    env.step(a)
torch.cuda.synchronize()
dt = time.time() - t0
print("BENCH " + json.dumps(dict(num_envs=args.num_envs, sensors=args.sensors, build_s=round(build_s, 1), env_steps_per_s=round(args.steps / dt, 2),
      samples_per_s=round(args.steps * args.num_envs / dt), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))), flush=True)
