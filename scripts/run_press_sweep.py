"""Controlled press sweep on the paper's in_fingers_rotate/Allegro env (no training).

Every env starts in the sampled grasp. After a hold phase, env e ramps the selected finger joints with
a constant action amp_e = e/(E-1) * amp_max for --press_steps (accumulated target), then holds. One
run therefore covers a whole range of press strengths in parallel. All envs are recorded; an env is
only valid until its first episode reset (object dropped).

Records, per env and fingertip: ground-truth |hand-object contact|, surface distance, and the taxel
reading of the chosen sensor (force_torque: sum |F|; elastomer: max displacement and loaded count).
Run once per sensor type; both runs use identical trajectories.

    python run_press_sweep.py --sensors actual-hand/elastomer --out sweep_el
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
ap.add_argument("--sensors", default="actual-hand/elastomer")
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--hold_steps", type=int, default=40)
ap.add_argument("--press_steps", type=int, default=80)
ap.add_argument("--settle_steps", type=int, default=40)
ap.add_argument("--amp_max", type=float, default=0.06)
ap.add_argument("--press_joints", default="bend")
ap.add_argument("--out", default="sweep")
args = ap.parse_args()

en.init(backend=gs.gpu, log_root_path="logs/tactile_record")
from registry import get_task_config

config = get_task_config(
    run_name="press_sweep",
    task_name=args.task,
    modifiers={"robot": args.robot, "sensors": args.sensors},
    config_override_path="conf/experiments/tiny.yaml",
)
config.env_options._locked = False
config.env_options.num_envs = args.num_envs
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
u = env.unwrapped
env.reset()
robot, obj = u.entities["robot"], u.entities["obj"]
E, n_act = args.num_envs, env.num_actions
kind = "elastomer" if "elastomer" in args.sensors else "force_torque"
TIPS = ["index", "middle", "ring", "thumb"]

names = [j.name for j in robot._entity.joints if j.n_dofs > 0]
sel = torch.zeros(n_act, device=gs.device)
keys = args.press_joints.split(",")
for i, n in enumerate(names[:n_act]):
    if any(k in n for k in keys):
        sel[i] = 1.0
print("press joints:", [n for n, s in zip(names, sel.tolist()) if s])
amps = torch.linspace(0, args.amp_max, E, device=gs.device)


def all_envs(x):
    x = x[0] if isinstance(x, tuple) and not hasattr(x, "force") else x
    x = x.force if hasattr(x, "force") else x
    return torch.as_tensor(x).float()


rec = {k: [] for k in ("gt", "dist", "tac_sum", "tac_max", "tac_loaded", "obj_z", "done")}
total = args.hold_steps + args.press_steps + args.settle_steps
for step in range(total):
    a = torch.zeros((E, n_act), device=gs.device)
    if args.hold_steps <= step < args.hold_steps + args.press_steps:
        a += amps[:, None] * sel[None, :]
    _, _, done, _ = env.step(a)
    gt, dist, ts, tm, tl = [], [], [], [], []
    for tip in TIPS:
        gt.append(all_envs(u.sensors[f"priv_contact_{tip}_3_tip"].read()).reshape(E, -1).norm(dim=-1))
        dist.append(all_envs(u.sensors[f"priv_surface_distance_{tip}_3_tip"].read()).reshape(E, -1)[:, 0])
        d = all_envs(u.sensors[f"tactile_{kind}_{tip}_3_tip"].read())
        d = d.reshape(E, -1, 3)
        h = d.shape[1] // 30
        d = d.reshape(E, h, 30, 3).median(dim=1).values
        m = d.norm(dim=-1)
        ts.append(m.sum(-1))
        tm.append(m.max(-1).values)
        tl.append((m > 1e-6).sum(-1))
    for k, v in zip(("gt", "dist", "tac_sum", "tac_max", "tac_loaded"), (gt, dist, ts, tm, tl)):
        rec[k].append(torch.stack(v, 1).cpu().numpy())
    rec["obj_z"].append(obj.get_pos().reshape(E, 3)[:, 2].cpu().numpy())
    rec["done"].append(done.reshape(E).cpu().numpy())

out = {k: np.stack(v) for k, v in rec.items()}
np.savez_compressed(f"{args.out}.npz", amps=amps.cpu().numpy(), dt=env.dt, hold=args.hold_steps,
                    press=args.press_steps, settle=args.settle_steps, kind=kind, tips=np.array(TIPS), **out)
first_done = np.where(out["done"].any(0), out["done"].argmax(0), -1)
print("saved", f"{args.out}.npz", "| first drop step per env:", first_done.tolist())
for e in range(E):
    s = slice(args.hold_steps + args.press_steps - 20, args.hold_steps + args.press_steps)
    print(f"env {e} amp {float(amps[e]):.3f} | plateau GT|F| per tip {np.round(out['gt'][s, e].mean(0), 2)} | tac_max {np.round(out['tac_max'][s, e].mean(0), 4)}")
