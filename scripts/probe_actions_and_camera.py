"""Print the hand's action DOF order, defaults and limits; try one headless camera render."""
import sys
import numpy as np
import torch
sys.path.insert(0, "src")
import genesis as gs
import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

en.init(backend=gs.gpu, log_root_path="logs/tactile_record")
from registry import get_task_config, position_camera_config

config = get_task_config(run_name="probe", task_name="in_fingers_rotate",
                         modifiers={"robot": "allegro", "sensors": "actual-hand/elastomer"},
                         config_override_path="conf/experiments/tiny.yaml")
config.env_options._locked = False
config.env_options.num_envs = 1
config = position_camera_config(config)
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
u = env.unwrapped
env.reset()
robot = u.entities["robot"]
names = [j.name for j in robot._entity.joints if j.n_dofs > 0]
lo, hi = robot.get_dofs_limit()
q = robot.get_dofs_pos()
for i, n in enumerate(names):
    print(f"{i:2d} {n:28s} q={float(q.reshape(-1)[i]): .3f} lo={float(lo.reshape(-1)[i]): .3f} hi={float(hi.reshape(-1)[i]): .3f}")
print("obj pos", u.entities["obj"].get_pos().tolist(), "robot pos", robot.get_pos().tolist())
rgb = u.cameras["rec"].render_rgb()
rgb = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
print("render ok", rgb.shape, rgb.dtype, float(rgb.mean()))
import cv2
cv2.imwrite("probe_frame.png", cv2.cvtColor(rgb.reshape(-1, *rgb.shape[-3:])[0][:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
