"""Drive the paper's own grasp_lift/Allegro env with a hand-written (non-learned) open-loop
descend-and-close sequence and record real tactile output.

grasp_lift starts the hand open and hovering over the object (no sampled grasp), and
RootPoseController maps actions to an ABSOLUTE wrist pose (0 = centre of each range), so an
all-zero action neither holds the hand still nor touches the object. This script commands the
wrist down and the finger flexors closed, then reports per-fingertip force/torque and distances.
Action layout: [0:16] finger targets, [16:22] wrist x,y,z,ex,ey,ez in [-1, 1].
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import genesis as gs
import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

en.init(backend=gs.gpu, log_root_path="logs/tactile_probe")

from registry import get_task_config
from tactile_record import TactileEpisodeRecorder

TABLE_Z = 0.5
WRIST_Z_RANGE = (TABLE_Z - 0.02, TABLE_Z + 0.50)

config = get_task_config(
    run_name="tactile_scripted",
    task_name="grasp_lift",
    modifiers={"robot": "allegro", "stage": 1, "sensors": "actual-hand/force_torque"},
    config_override_path="conf/experiments/tiny.yaml",
)
config.env_options._locked = False
config.env_options.num_envs = 1
env = RslRlVecEnvWrapper.from_config(config, show_viewer=False)
u = env.unwrapped
obs, _ = env.reset()

robot, obj = u.entities["robot"], u.entities["obj"]
joint_names = [j.name for j in robot._entity.joints if j.n_dofs > 0]
print("JOINTS:", joint_names)
hand_names = [n for n in joint_names if ("bend" in n or "roll" in n)]
print("HAND JOINTS (action order assumed):", hand_names, len(hand_names))
print("RESET obj pos:", obj.get_pos().tolist(), " robot pos:", robot.get_pos().tolist())

TIPS = ["index", "middle", "ring", "thumb"]
tip_force = {t: f"tactile_force_torque_{t}_3_tip" for t in TIPS}
tip_dist = {t: f"priv_surface_distance_{t}_3_tip" for t in TIPS}


def read_min_dist(name):
    d = u.sensors[name].read()
    d = d[0] if isinstance(d, tuple) else d
    return float(torch.as_tensor(d).float().min())


def wrist_action(z_target):
    center = 0.5 * (WRIST_Z_RANGE[0] + WRIST_Z_RANGE[1])
    half = 0.5 * (WRIST_Z_RANGE[1] - WRIST_Z_RANGE[0])
    return float(np.clip((z_target - center) / half, -1.0, 1.0))


def finger_action(sign, closure):
    a = torch.zeros(16, device=gs.device)
    for i, n in enumerate(hand_names):
        if "bend" in n:
            a[i] = sign * closure
    return a


def _rng(ent):
    e = ent._entity
    return e.link_start, e.link_start + e.n_links


ROBOT_R = _rng(robot)
OBJ_R = _rng(obj)
TABLE_R = _rng(u.entities["table"]) if "table" in u.entities else (-1, -1)
TIP_LINK = {t: robot._entity.get_link(f"{t}_3_tip").idx for t in TIPS}
print("ENTITIES:", list(u.entities.keys()), "link ranges robot/obj/table:", ROBOT_R, OBJ_R, TABLE_R)


def _in(r, i):
    return r[0] <= i < r[1]


def contact_breakdown():
    """Ground-truth rigid-solver contacts of the hand, bucketed by partner, plus per-tip force on obj."""
    c = robot._entity.get_contacts()
    la, lb = c["link_a"].reshape(-1), c["link_b"].reshape(-1)
    fa = c["force_a"].reshape(-1, 3).norm(dim=-1)
    ok = c["valid_mask"].reshape(-1) if "valid_mask" in c else torch.ones_like(fa, dtype=torch.bool)
    out = {"obj": 0.0, "table": 0.0, "self": 0.0, "other": 0.0}
    tip_obj = {t: 0.0 for t in TIPS}
    tip_cat = {t: {"obj": 0.0, "self": 0.0, "table": 0.0} for t in TIPS}
    for a, b, f, v in zip(la.tolist(), lb.tolist(), fa.tolist(), ok.tolist()):
        if not v:
            continue
        ra, rb = _in(ROBOT_R, a), _in(ROBOT_R, b)
        if ra and rb:
            out["self"] = max(out["self"], f)
            for t, li in TIP_LINK.items():
                if li in (a, b):
                    tip_cat[t]["self"] += f
            continue
        hand_l, other = (a, b) if ra else (b, a)
        key = "obj" if _in(OBJ_R, other) else "table" if _in(TABLE_R, other) else "other"
        out[key] = max(out[key], f)
        for t, li in TIP_LINK.items():
            if hand_l == li and key in tip_cat[t]:
                tip_cat[t][key] += f
        if key == "obj":
            for t, li in TIP_LINK.items():
                if hand_l == li:
                    tip_obj[t] += f
    return out, tip_obj, tip_cat


def run_trial(depth, sign, n_descend=40, n_close=60, n_hold=20):
    obs, _ = env.reset()
    z0 = float(robot.get_pos()[0, 2]) if robot.get_pos().ndim > 1 else float(robot.get_pos()[2])
    rec = TactileEpisodeRecorder(u, robot="allegro", task="grasp_lift")
    peak = {t: 0.0 for t in TIPS}
    peak_any = 0.0
    min_d = {t: 1e9 for t in TIPS}
    obj_xyz = obj.get_pos().reshape(-1, 3)[0].cpu()
    obj_z0 = float(obj_xyz[2])
    gt_peak = {"obj": 0.0, "table": 0.0, "self": 0.0, "other": 0.0}
    gt_tip_obj = {t: 0.0 for t in TIPS}
    trace = []
    t = 0.0
    total = n_descend + n_close + n_hold
    for step in range(total):
        a = torch.zeros((1, 22), device=gs.device)
        z_target = z0 - depth * min(1.0, (step + 1) / n_descend)
        a[0, 16] = float(np.clip(obj_xyz[0] / 0.15, -1, 1))
        a[0, 17] = float(np.clip(obj_xyz[1] / 0.15, -1, 1))
        a[0, 18] = wrist_action(z_target)
        closure = 0.0 if step < n_descend else min(1.0, (step - n_descend + 1) / n_close)
        a[0, :16] = finger_action(sign, closure)
        env.step(a)
        t += env.dt
        rec.record(t)
        for tip in TIPS:
            f = np.linalg.norm(rec._data[tip_force[tip]][-1], axis=-1).max()
            peak[tip] = max(peak[tip], float(f))
            min_d[tip] = min(min_d[tip], read_min_dist(tip_dist[tip]))
        cb, tb, tc = contact_breakdown()
        for k in gt_peak:
            gt_peak[k] = max(gt_peak[k], cb[k])
        for k in gt_tip_obj:
            gt_tip_obj[k] = max(gt_tip_obj[k], tb[k])
        tn = "thumb"
        vecs = rec._data[tip_force[tn]][-1]
        trace.append(
            (
                step,
                closure,
                float(np.linalg.norm(vecs.sum(axis=0))),
                float(np.linalg.norm(vecs, axis=-1).max()),
                int((np.linalg.norm(vecs, axis=-1) > 1e-6).sum()),
                tc[tn]["obj"],
                tc[tn]["self"],
                tc[tn]["table"],
                read_min_dist(tip_dist[tn]),
            )
        )
        for name in rec.sensor_names:
            peak_any = max(peak_any, float(np.linalg.norm(rec._data[name][-1], axis=-1).max()))
    obj_dz = float(obj.get_pos().reshape(-1, 3)[0, 2]) - obj_z0
    print("   THUMB TRACE  step closure | taxel sum|F| max|F| n_loaded | GT obj self table | surf_dist")
    for row in [r for r in trace if r[0] % 5 == 0 or r[0] >= total - 20]:
        print("   %4d  %.2f | %.4f %.4f %3d | %.3f %.3f %.3f | %.4f" % row)
    tag = f"d{int(depth*1000)}mm_s{'p' if sign > 0 else 'n' if sign < 0 else '0'}"
    path = rec.save(f"scripted_{tag}.npz")
    print(
        f"TRIAL depth={depth:.3f} sign={sign:+d} | peak|F| per tip "
        + " ".join(f"{t}={peak[t]:.4f}" for t in TIPS)
        + f" | any-sensor peak={peak_any:.4f} | min surf dist "
        + " ".join(f"{t}={min_d[t]:.4f}" for t in TIPS)
        + f" | obj dz={obj_dz:+.4f} | saved {path}\n"
        + "      GT solver peak |F| by partner: "
        + " ".join(f"{k}={v:.4f}" for k, v in gt_peak.items())
        + " | GT tip<->obj: "
        + " ".join(f"{t}={gt_tip_obj[t]:.4f}" for t in TIPS),
        flush=True,
    )


for depth in (0.01,):
    run_trial(depth, +1)
print("SCRIPTED DONE")
