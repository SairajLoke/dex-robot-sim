"""Summarise a run_task_tactile_record.py recording: taxel force vs privileged ground-truth contact.

    python scripts/plot_task_tactile_record.py rec_ft --hold_steps 100
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("stem")
ap.add_argument("--hold_steps", type=int, default=100)
ap.add_argument("--kind", default="force_torque")
args = ap.parse_args()


def reduce_history(d, name):
    """(T, H*N, 3) -> (T, N, 3): the recorder flattens the H sub-step history entries (history-major)."""
    x = d[f"{name}__data"]
    n = d[f"{name}__pos"].shape[1]
    h = x.shape[1] // n
    return np.median(x.reshape(x.shape[0], h, n, 3), axis=1)

d = np.load(f"{args.stem}.npz", allow_pickle=True)
e = np.load(f"{args.stem}_extra.npz")
t = e["t"]
tips = ["index", "middle", "ring", "thumb"]
done = np.nonzero(e["done"])[0]

fig, ax = plt.subplots(len(tips) + 2, 1, figsize=(11, 12), sharex=True)
print(f"{'tip':7s} {'corr(taxel,GT) all':>19s} {'hold':>7s}  {'taxel sum|F| hold':>18s} {'GT |F| hold':>12s}")
for i, tip in enumerate(tips):
    f = np.linalg.norm(reduce_history(d, f"tactile_{args.kind}_{tip}_3_tip"), axis=-1)
    tot, nl = f.sum(-1), (f > 1e-6).sum(-1)
    gt = np.linalg.norm(e[f"priv_contact_{tip}_3_tip"], axis=-1)
    c_all = np.corrcoef(tot, gt)[0, 1] if tot.std() > 0 and gt.std() > 0 else float("nan")
    h = slice(0, args.hold_steps)
    c_h = np.corrcoef(tot[h], gt[h])[0, 1] if tot[h].std() > 0 and gt[h].std() > 0 else float("nan")
    print(f"{tip:7s} {c_all:19.3f} {c_h:7.3f}  {tot[h].mean():18.3f} {gt[h].mean():12.3f}")
    a = ax[i]
    a.plot(t, tot, color="C0", label="taxel sum |F|")
    a.set_ylabel(f"{tip}\ntaxel sum|F|", color="C0")
    b = a.twinx()
    b.plot(t, gt, color="C3", alpha=0.7, label="GT |contact|")
    b.plot(t, nl / 10.0, color="C2", alpha=0.5, label="loaded taxels / 10")
    b.set_ylabel("GT |contact|", color="C3")
    if i == 0:
        a.legend(loc="upper left", fontsize=7)
        b.legend(loc="upper right", fontsize=7)
for tip in tips:
    ax[-2].plot(t, e[f"priv_surface_distance_{tip}_3_tip"][:, 0], label=tip)
ax[-2].set_ylabel("surface dist (m)")
ax[-2].legend(ncol=4, fontsize=7)
ax[-1].plot(t, e["obj_pos"][:, 2], label="object z")
ax[-1].set_ylabel("object z (m)")
rb = ax[-1].twinx()
rb.plot(t, e["reward"], color="C1", alpha=0.6)
rb.set_ylabel("reward", color="C1")
for a in ax:
    a.axvline(t[args.hold_steps - 1], color="k", ls="--", lw=0.8)
    for k in done:
        a.axvline(t[k], color="r", ls=":", lw=0.8)
ax[-1].set_xlabel("time (s)   dashed = hold -> wiggle, red dotted = episode reset (object dropped)")
fig.suptitle(f"{args.stem}: {d['task']} / {d['robot']} -- tactile {args.kind} vs ground truth")
fig.tight_layout()
fig.savefig(f"{args.stem}_summary.png", dpi=110)
print("saved", f"{args.stem}_summary.png")
