"""Controlled single-fingertip press: slow target ramp of the middle finger into a fixed sphere.

Same scene as hand_object_tactile_demo.py (paper's Allegro asset, real 368-taxel fingertip layout, fixed
sphere, no gravity) but only the middle finger moves, ramping its bend targets well past first contact
so the PD controller drives the contact force up gradually. Each step records, for the middle fingertip:
ground-truth rigid-solver contact force on the sphere (summed), KinematicTaxel force/torque, and
ElastomerTaxel displacement. Output: press_single_<tag>.npz.

    python press_single_fingertip.py --bend_end 1.2 --out press_single
"""

import argparse
import json
import sys

import yaml

import numpy as np
import torch

sys.path.insert(0, "src")
import genesis as gs
import genesis.utils.geom as gu
from entities.robots.allegro_hand import AllegroHand

ap = argparse.ArgumentParser()
ap.add_argument("--bend_start", type=float, default=0.3)
ap.add_argument("--bend_end", type=float, default=1.2)
ap.add_argument("--ramp_steps", type=int, default=450)
ap.add_argument("--obj_pos", type=float, nargs=3, default=(0.123, 0.0, 0.233))
ap.add_argument("--radius", type=float, default=0.075)
ap.add_argument("--n_sample_points", type=int, default=None, help="override the paper value (1000)")
ap.add_argument("--ghost", action="store_true", help="sphere has no collision: probes can penetrate it (sensor-only test)")
ap.add_argument("--out", default="press_single")
args = ap.parse_args()

PARAMS = yaml.safe_load(open("conf/sensor/tactile_params.yaml"))
EP = {k: v for k, v in PARAMS["elastomer"]["params"].items() if not k.startswith("debug_")}
FP = {k: v for k, v in PARAMS["force_torque"]["params"].items() if not k.startswith("debug_")}
if args.n_sample_points:
    EP["n_sample_points"] = args.n_sample_points
print("elastomer params:", EP, "\nforce_torque params:", FP)
PROBE_JSON = "src/assets/sensors/allegro/actual/probes_368_hand_allegro.json"
TIP = "middle_3_tip"
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)), show_viewer=False)
spec = AllegroHand()
hand = scene.add_entity(gs.morphs.MJCF(file=spec.file, pos=(0.0, 0.0, 0.2)))
if args.ghost:
    hand._is_local_collision_mask = False  # let the sphere's disjoint contype/conaffinity mask filter hand<->sphere
obj = scene.add_entity(gs.morphs.Sphere(radius=args.radius, pos=tuple(args.obj_pos), fixed=True, **({'contype': 0x10000, 'conaffinity': 0x10000} if args.ghost else {})))

entry = next(e for e in json.load(open(PROBE_JSON)) if e["link_name"] == TIP)
flat = [p for row in entry["probes"] for p in row] if entry["kind"] == "grid" else entry["probes"]
pos = np.array([p["pos"] for p in flat], dtype=np.float64)
normal = np.array([p["normal"] for p in flat], dtype=np.float64)
radius = np.array([p["radius"] for p in flat], dtype=np.float64)
link = hand.get_link(TIP)
hand_links = tuple(range(hand.link_start, hand.link_start + hand.n_links))
kin = scene.add_sensor(gs.sensors.KinematicTaxel(
    entity_idx=hand.idx, link_idx_local=link.idx_local, probe_local_pos=pos, probe_radius=radius,
    filter_link_idx=hand_links, **FP))
ela = scene.add_sensor(gs.sensors.ElastomerTaxel(
    entity_idx=hand.idx, link_idx_local=link.idx_local, probe_local_pos=pos, probe_local_normal=normal,
    probe_radius=radius, track_link_idx=(obj.base_link_idx,), **EP))
scene.build(n_envs=0)

names = spec.dofs_name
idx = np.array([hand.get_joint(n).dofs_idx_local[0] for n in names])
default = np.array([spec.default_dofs_pos[n] for n in names])
hand.set_dofs_position(default, idx)
hand.control_dofs_position(default, idx)
mid = [names.index(f"middle_bend{i}") for i in range(3)]
tip_idx, obj_idx = link.idx, obj.base_link_idx

probe_local = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
radius_t = torch.as_tensor(radius, dtype=gs.tc_float, device=gs.device)
center = torch.as_tensor(args.obj_pos, dtype=gs.tc_float, device=gs.device)
rows = []
for i in range(40 + args.ramp_steps):
    tgt = default.copy()
    if i >= 40:
        b = args.bend_start + (args.bend_end - args.bend_start) * (i - 40) / (args.ramp_steps - 1)
        tgt[mid] = b
    hand.control_dofs_position(tgt, idx)
    scene.step()
    c = hand.get_contacts()
    la, lb = c["link_a"].reshape(-1), c["link_b"].reshape(-1)
    f = c["force_a"].reshape(-1, 3).norm(dim=-1)
    ok = c["valid_mask"].reshape(-1) if "valid_mask" in c else torch.ones_like(f, dtype=torch.bool)
    sel = ok & (((la == tip_idx) & (lb == obj_idx)) | ((lb == tip_idx) & (la == obj_idx)))
    gt = float(f[sel].sum())
    pw = gu.transform_by_trans_quat(probe_local, link.get_pos().reshape(3), link.get_quat().reshape(4))
    sd = (pw - center).norm(dim=-1) - args.radius
    kd = kin.read_ground_truth()
    el = ela.read_ground_truth()
    fm, em = torch.linalg.norm(kd.force, dim=-1), torch.linalg.norm(el, dim=-1)
    rows.append((i, float(tgt[mid[0]]), gt, float(fm.sum()), float(fm.max()), float(torch.linalg.norm(kd.torque, dim=-1).max()),
                 int((fm > 1e-6).sum()), float(em.max()), float(em.sum()), int((em > 1e-9).sum()), float(sd.min()), float((sd - radius_t).min())))
a = np.array(rows)
np.savez_compressed(f"{args.out}.npz", cols=np.array(["step", "bend_target", "gt_force", "ft_sum", "ft_max", "ft_torque_max", "ft_n", "el_max", "el_sum", "el_n", "probe_min_sd", "probe_min_sd_minus_r"]), data=a)
print("saved", f"{args.out}.npz", a.shape)
for r in a[::25]:
    print("step %3d tgt %.2f | GT %.3f N | ft sum %.3f max %.3f n=%2d | el max %.5f sum %.5f n=%2d | min probe sd %.4f m" % (r[0], r[1], r[2], r[3], r[4], r[6], r[7], r[8], r[9], r[10]))
print("probe signed dist to sphere surface (m, <0 = probe inside object): min over run", a[:, 10].min(), "| at final step", a[-1, 10], "| min(sd - probe radius)", a[:, 11].min())
m = a[:, 2] > 1e-6
print("steps with GT contact:", int(m.sum()), "| max GT", a[:, 2].max(), "| max el", a[:, 7].max(), "| corr(GT, el_max) in contact:", np.corrcoef(a[m, 2], a[m, 7])[0, 1] if m.sum() > 2 and a[m, 7].std() > 0 else "n/a")
