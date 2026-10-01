"""Animate the four fingertip tactile pads of a run_task_tactile_record.py recording.

Left: 3D view of every taxel (colour = |reading|) plus the object centre; right: per-tip total
reading over time with a cursor. Standalone because scripts/animate_tactile.py fails on this
recording (palm sensor probe-position vs data shape mismatch).

    python scripts/animate_task_tactile.py rec_ft --kind force_torque --fps 20
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("stem")
ap.add_argument("--kind", default="force_torque")
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--dpi", type=int, default=80)
args = ap.parse_args()


def reduce_history(d, name):
    """(T, H*N, 3) -> (T, N, 3): the recorder flattens the H sub-step history entries (history-major)."""
    x = d[f"{name}__data"]
    n = d[f"{name}__pos"].shape[1]
    h = x.shape[1] // n
    return np.median(x.reshape(x.shape[0], h, n, 3), axis=1)

d = np.load(f"{args.stem}.npz", allow_pickle=True)
e = np.load(f"{args.stem}_extra.npz")
tips = ["index", "middle", "ring", "thumb"]
pos = np.concatenate([d[f"tactile_{args.kind}_{t}_3_tip__pos"] for t in tips], axis=1)
mag = np.concatenate([np.linalg.norm(reduce_history(d, f"tactile_{args.kind}_{t}_3_tip"), axis=-1) for t in tips], axis=1)
tot = np.stack([np.linalg.norm(reduce_history(d, f"tactile_{args.kind}_{t}_3_tip"), axis=-1).sum(-1) for t in tips], 1)
t = d["t"]
vmax = max(float(np.percentile(mag, 99.5)), 1e-9)
centre = pos.reshape(-1, 3).mean(0)
half = 0.06
done = set(np.nonzero(e["done"])[0].tolist())

fig = plt.figure(figsize=(11, 5))
a3 = fig.add_subplot(1, 2, 1, projection="3d")
at = fig.add_subplot(1, 2, 2)
sc = a3.scatter([], [], [], c=[], cmap="inferno", vmin=0, vmax=vmax, s=14)
ob = a3.scatter([], [], [], c="tab:cyan", s=120, marker="o", alpha=0.6)
a3.set_xlim(centre[0] - half, centre[0] + half)
a3.set_ylim(centre[1] - half, centre[1] + half)
a3.set_zlim(centre[2] - half, centre[2] + half)
a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z")
fig.colorbar(sc, ax=a3, shrink=0.6, label=f"|{args.kind}|")
for i, tip in enumerate(tips):
    at.plot(t, tot[:, i], label=tip)
at.set_xlabel("time (s)"); at.set_ylabel(f"sum |{args.kind}| per tip"); at.legend(fontsize=8)
cur = at.axvline(t[0], color="k")
title = fig.suptitle("")


def update(k):
    sc._offsets3d = (pos[k, :, 0], pos[k, :, 1], pos[k, :, 2])
    sc.set_array(mag[k])
    ob._offsets3d = ([e["obj_pos"][k, 0]], [e["obj_pos"][k, 1]], [e["obj_pos"][k, 2]])
    cur.set_xdata([t[k], t[k]])
    title.set_text(f"{d['task']} / {d['robot']}  t={t[k]:.2f}s" + ("   [object dropped -> reset]" if k in done else ""))
    return sc, ob, cur


ani = FuncAnimation(fig, update, frames=len(t), interval=1000 / args.fps)
out = f"{args.stem}_tips.mp4"
ani.save(out, writer=FFMpegWriter(fps=args.fps), dpi=args.dpi)
print("saved", out)
