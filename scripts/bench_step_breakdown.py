"""Time raw scene.step() vs the full env.step() (managers, rewards, obs) for the paper's in_fingers_rotate env."""
import argparse, json, sys, time
import torch
sys.path.insert(0, "src")
import genesis as gs
import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=1024)
args = ap.parse_args()
en.init(backend=gs.gpu, log_root_path="logs/bench")
from registry import get_task_config

config = get_task_config(run_name="bench", task_name="in_fingers_rotate", modifiers={"robot": "allegro"}, config_override_path="conf/experiments/tiny.yaml")
config.env_options._locked = False
config.env_options.num_envs = args.num_envs
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
env.reset()
u = env.unwrapped
a = torch.zeros((args.num_envs, env.num_actions), device=gs.device)
for _ in range(10):
    env.step(a)
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    env.step(a)
torch.cuda.synchronize()
t_env = (time.time() - t) / 20
t = time.time()
for _ in range(20):
    u.scene.step()
torch.cuda.synchronize()
t_scene = (time.time() - t) / 20
print("BREAKDOWN " + json.dumps(dict(num_envs=args.num_envs, env_step_s=round(t_env, 4), scene_step_s=round(t_scene, 4), decimation_note="env.step runs several scene.step()s per call", substeps=getattr(u, "decimation", None))))
