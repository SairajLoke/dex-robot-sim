import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import genesis as gs
import genesis.utils.geom as gu
from entities.robots.allegro_hand import AllegroHand

PROBE_JSON = "src/assets/sensors/allegro/actual/probes_368_hand_allegro.json"
FINGERTIPS = ["index_3_tip", "middle_3_tip", "ring_3_tip"]

# Known achieved equilibrium under control_dofs_position(target=1.1) for all three bend
# joints per finger, from hand_demo4/5/6 logs (self-collision stalls bend1/bend2 early
# while bend0 overshoots -- this triple is what the controller actually converges to,
# regardless of nearby object placement).
ACHIEVED = {"bend0": 0.6, "bend1": 0.6, "bend2": 0.6}


def load_fingertip_probes():
    with open(PROBE_JSON) as f:
        entries = json.load(f)
    by_link = {}
    for e in entries:
        if e["link_name"] not in FINGERTIPS:
            continue
        probes = e["probes"]
        flat = [p for row in probes for p in row] if e["kind"] == "grid" else probes
        by_link[e["link_name"]] = np.array([p["pos"] for p in flat], dtype=np.float64)
    return by_link


gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)), show_viewer=False)
hand_spec = AllegroHand()
hand = scene.add_entity(gs.morphs.MJCF(file=hand_spec.file, pos=(0.0, 0.0, 0.2)))
scene.build(n_envs=0)

dofs_name = hand_spec.dofs_name
dofs_idx = np.array([hand.get_joint(n).dofs_idx_local[0] for n in dofs_name])
default_pos = np.array([hand_spec.default_dofs_pos[n] for n in dofs_name])
q = default_pos.copy()
for finger in ("index", "middle", "ring"):
    for part in ("bend0", "bend1", "bend2"):
        q[dofs_name.index(f"{finger}_{part}")] = ACHIEVED[part]

hand.set_dofs_position(q, dofs_idx, zero_velocity=True)
scene.step()

probes = load_fingertip_probes()
all_world = []
for link_name in FINGERTIPS:
    link = hand.get_link(link_name)
    pos = hand.get_links_pos(link.idx_local).cpu().numpy().reshape(3)
    quat = hand.get_links_quat(link.idx_local).cpu().numpy().reshape(4)  # w,x,y,z
    local = probes[link_name]
    world = gu.transform_by_quat(torch.as_tensor(local, dtype=torch.float64), torch.as_tensor(quat, dtype=torch.float64)).cpu().numpy() + pos
    all_world.append(world)
    centroid = world.mean(axis=0)
    extent = np.linalg.norm(world - centroid, axis=1).max()
    print(f"{link_name}: link_pos={pos}  taxel_centroid={centroid}  max_extent={extent:.4f}")

combined = np.concatenate(all_world, axis=0)
combined_centroid = combined.mean(axis=0)
combined_radius = np.linalg.norm(combined - combined_centroid, axis=1).max()
print(f"\ncombined centroid={combined_centroid}  radius_needed={combined_radius:.4f}")
