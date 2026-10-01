"""
Allegro hand grasping a sphere, reading real per-fingertip torque (KinematicTaxel) and
tactile deformation (ElastomerTaxel) from Genesis as the fingers close onto the object.

Uses the paper's own Allegro asset, DOF ordering, grasp_center, and the real 368-probe
fingertip taxel layout (src/assets/sensors/allegro/actual/probes_368_hand_allegro.json),
restricted to the 4 fingertip links. Not the paper's RL task/policy stack -- this is a
scripted open/close to directly exercise the sensors against a real hand + real object,
headless, logging CSV + a summary plot.
"""

import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import genesis as gs
from entities.robots.allegro_hand import AllegroHand

PROBE_JSON = "src/assets/sensors/allegro/actual/probes_368_hand_allegro.json"
FINGERTIPS = ["index_3_tip", "middle_3_tip", "ring_3_tip", "thumb_3_tip"]
N_SETTLE = 40
N_CLOSE = 160
N_HOLD = 40
DT = 0.01

# Per-finger closing targets (bend joints only; roll stays at default). Values are within
# each joint's MJCF ctrlrange (see right_hand.xml), picked to curl the fingers toward the
# grasp_center sphere without self-colliding.
CLOSE_TARGETS = {
    "index_bend0": 0.95,
    "index_bend1": 1.0,
    "index_bend2": 0.9,
    "middle_bend0": 0.95,
    "middle_bend1": 1.0,
    "middle_bend2": 0.9,
    "ring_bend0": 0.95,
    "ring_bend1": 1.0,
    "ring_bend2": 0.9,
    "thumb_roll": 0.55,
    "thumb_bend1": 1.0,
    "thumb_bend2": 0.9,
}


def load_fingertip_probes():
    with open(PROBE_JSON) as f:
        entries = json.load(f)
    by_link = {}
    for e in entries:
        if e["link_name"] not in FINGERTIPS:
            continue
        probes = e["probes"]
        flat = [p for row in probes for p in row] if e["kind"] == "grid" else probes
        pos = np.array([p["pos"] for p in flat], dtype=np.float64)
        normal = np.array([p["normal"] for p in flat], dtype=np.float64)
        radius = np.array([p["radius"] for p in flat], dtype=np.float64)
        by_link[e["link_name"]] = (pos, normal, radius)
    return by_link


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        show_viewer=False,
    )

    hand_spec = AllegroHand()
    hand = scene.add_entity(
        gs.morphs.MJCF(file=hand_spec.file, pos=(0.0, 0.0, 0.2)),
    )

    # grasp_center is defined in the palm-link local frame; with the hand at identity
    # orientation this is also the offset from the hand's base position.
    gx, gy, gz = hand_spec.metadata.grasp_center
    hx, hy, hz = 0.0, 0.0, 0.2
    object_pos = (hx + gx, hy + gy, hz + gz)
    SPHERE_RADIUS = 0.028
    obj = scene.add_entity(
        gs.morphs.Sphere(radius=SPHERE_RADIUS, pos=object_pos, fixed=False),
    )

    fingertip_probes = load_fingertip_probes()
    kin_sensors, elast_sensors = {}, {}
    for link_name in FINGERTIPS:
        pos, normal, radius = fingertip_probes[link_name]
        link = hand.get_link(link_name)
        kin_sensors[link_name] = scene.add_sensor(
            gs.sensors.KinematicTaxel(
                entity_idx=hand.idx,
                link_idx_local=link.idx_local,
                probe_local_pos=pos,
                probe_radius=radius,
                normal_stiffness=500.0,
                normal_damping=1.0,
                shear_scalar=4.0,
                twist_scalar=1.0,
            )
        )
        elast_sensors[link_name] = scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                entity_idx=hand.idx,
                link_idx_local=link.idx_local,
                probe_local_pos=pos,
                probe_local_normal=normal,
                probe_radius=radius,
                track_link_idx=(obj.base_link_idx,),
                n_sample_points=600,
                lambda_d=5000.0,
                lambda_s=4000.0,
                dilate_scale=0.1,
                shear_scale=1.0,
                normal_exponent=1.5,
                compressibility=0.8,
            )
        )

    scene.build(n_envs=0)

    print("DEBUG object_pos:", object_pos)
    for ln in FINGERTIPS:
        link = hand.get_link(ln)
        print("DEBUG", ln, hand.get_links_pos(link.idx_local).cpu().numpy())
    print("DEBUG hand base pos:", hand.get_links_pos(hand.base_link_idx).cpu().numpy())


    dofs_name = hand_spec.dofs_name
    dofs_idx = np.array([hand.get_joint(n).dofs_idx_local[0] for n in dofs_name])
    default_pos = np.array([hand_spec.default_dofs_pos[n] for n in dofs_name])
    close_pos = default_pos.copy()
    for name, val in CLOSE_TARGETS.items():
        close_pos[dofs_name.index(name)] = val

    hand.set_dofs_position(default_pos, dofs_idx)
    hand.control_dofs_position(default_pos, dofs_idx)

    rows = []

    def record(step, phase):
        row = {"step": step, "phase": phase}
        for name in FINGERTIPS:
            kd = kin_sensors[name].read_ground_truth()
            force_mag = torch.linalg.norm(kd.force, dim=-1)
            torque_mag = torch.linalg.norm(kd.torque, dim=-1)
            elast = elast_sensors[name].read_ground_truth()
            elast_mag = torch.linalg.norm(elast, dim=-1)
            row[f"{name}_force_max"] = force_mag.max().item()
            row[f"{name}_force_sum"] = force_mag.sum().item()
            row[f"{name}_torque_max"] = torque_mag.max().item()
            row[f"{name}_deform_max"] = elast_mag.max().item()
            row[f"{name}_deform_sum"] = elast_mag.sum().item()
            row[f"{name}_n_active"] = int((force_mag > 1e-6).sum().item())
        rows.append(row)

    for i in range(N_SETTLE):
        scene.step()
        if i % 5 == 0:
            record(i, "settle")

    for i in range(N_CLOSE):
        alpha = (i + 1) / N_CLOSE
        target = default_pos + alpha * (close_pos - default_pos)
        hand.control_dofs_position(target, dofs_idx)
        scene.step()
        if i % 4 == 0:
            record(N_SETTLE + i, "close")

    for i in range(N_HOLD):
        hand.control_dofs_position(close_pos, dofs_idx)
        scene.step()
        if i % 4 == 0:
            record(N_SETTLE + N_CLOSE + i, "hold")

    out_csv = "hand_object_tactile_demo.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_csv}\n")
    print(f"{'finger':<14}{'max_force':>12}{'max_torque':>12}{'max_deform':>12}{'max_taxels_active':>20}")
    for name in FINGERTIPS:
        mf = max(r[f"{name}_force_max"] for r in rows)
        mt = max(r[f"{name}_torque_max"] for r in rows)
        md = max(r[f"{name}_deform_max"] for r in rows)
        ma = max(r[f"{name}_n_active"] for r in rows)
        print(f"{name:<14}{mf:>12.5f}{mt:>12.6f}{md:>12.6f}{ma:>20d}")

    settle_force = max(
        r[f"{name}_force_max"] for r in rows if r["phase"] == "settle" for name in FINGERTIPS
    )
    print(f"\nMax force magnitude during settle phase (no contact expected): {settle_force:.3e}")


if __name__ == "__main__":
    main()
