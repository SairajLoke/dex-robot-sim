import sys
import numpy as np

sys.path.insert(0, "src")
import genesis as gs
from entities.robots.allegro_hand import AllegroHand

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)), show_viewer=False)
hand_spec = AllegroHand()
hand = scene.add_entity(gs.morphs.MJCF(file=hand_spec.file, pos=(0.0, 0.0, 0.2)))
scene.build(n_envs=0)

dofs_name = hand_spec.dofs_name
dofs_idx = np.array([hand.get_joint(n).dofs_idx_local[0] for n in dofs_name])
default_pos = np.array([hand_spec.default_dofs_pos[n] for n in dofs_name])

FINGERTIPS = ["index_3_tip", "middle_3_tip", "ring_3_tip", "thumb_3_tip"]
links = {n: hand.get_link(n) for n in FINGERTIPS}


def report(label, qpos):
    hand.set_dofs_position(qpos, dofs_idx, zero_velocity=True)
    scene.step()
    print(f"\n-- {label} --")
    for n in FINGERTIPS:
        p = hand.get_links_pos(links[n].idx_local).cpu().numpy().reshape(3)
        print(f"  {n:<14} {p}")


report("default (all 0, thumb_bend0=0.263)", default_pos.copy())

for bend in (0.5, 1.0, 1.5):
    q = default_pos.copy()
    for name in ("index_bend0", "index_bend1", "index_bend2",
                 "middle_bend0", "middle_bend1", "middle_bend2",
                 "ring_bend0", "ring_bend1", "ring_bend2"):
        q[dofs_name.index(name)] = bend
    report(f"index/middle/ring bend={bend}", q)

for thumb_bend0 in (0.263, 0.6, 1.0, 1.396):
    for thumb_roll in (0.0, 0.5, 1.0):
        q = default_pos.copy()
        q[dofs_name.index("thumb_bend0")] = thumb_bend0
        q[dofs_name.index("thumb_roll")] = thumb_roll
        q[dofs_name.index("thumb_bend1")] = 1.0
        q[dofs_name.index("thumb_bend2")] = 1.0
        report(f"thumb_bend0={thumb_bend0} thumb_roll={thumb_roll} bend1/2=1.0", q)

print("\nhand base pos:", hand.get_links_pos(hand.base_link_idx).cpu().numpy())
