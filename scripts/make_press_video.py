"""Compose the rendered interaction video with synchronized tactile / ground-truth time series.

Inputs (from run_task_tactile_record.py --squeeze_steps N --video, run once per sensor type):
  <ft>.npz/_extra.npz  force_torque run     <el>.npz/_extra.npz  elastomer run
  <raw>.mp4            rendered frames of the elastomer run (identical trajectory to the ft run)

    python scripts/make_press_video.py --ft docs/data/press_ft --el docs/data/press_el --raw docs/data/press_el_raw.mp4 --out docs/data/press_interaction.mp4
"""

import argparse
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ft", required=True)
ap.add_argument("--el", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
TIPS = ["index", "middle", "ring", "thumb"]
COL = dict(zip(TIPS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))


def reduce_history(d, name, T):
    a = d[f"{name}__data"]
    H = 5
    return np.median(a.reshape(T, H, a.shape[1] // H, 3), axis=1)


ft, el = np.load(args.ft + ".npz", allow_pickle=True), np.load(args.el + ".npz", allow_pickle=True)
ex = np.load(args.el + "_extra.npz")
t, ph, T = ex["t"], ex["phase"], len(ex["t"])
ft_sum = {k: np.linalg.norm(reduce_history(ft, f"tactile_force_torque_{k}_3_tip", T), axis=-1).sum(1) for k in TIPS}
el_max = {k: np.linalg.norm(reduce_history(el, f"tactile_elastomer_{k}_3_tip", T), axis=-1).max(1) * 1e3 for k in TIPS}
gt = {k: np.linalg.norm(ex[f"priv_contact_{k}_3_tip"], axis=-1) for k in TIPS}

W, Hh = 800, 720
fig, axs = plt.subplots(4, 1, figsize=(W / 100, Hh / 100), dpi=100, sharex=True)
fig.subplots_adjust(left=0.11, right=0.97, top=0.96, bottom=0.07, hspace=0.18)
for ax, (title, data) in zip(axs[:3], [("GT hand-object |F| (N)", gt), ("force_torque taxel sum |F|", ft_sum), ("elastomer max displacement (mm)", el_max)]):
    for k in TIPS:
        ax.plot(t, data[k], color=COL[k], lw=1, label=k)
    ax.set_ylabel(title, fontsize=7)
axs[0].legend(ncol=4, fontsize=7, loc="upper right")
axs[3].plot(t, ex["obj_pos"][:, 2], color="k", lw=1)
axs[3].set_ylabel("object z (m)", fontsize=7)
axs[3].set_xlabel("time (s)", fontsize=8)
bounds = [i for i in range(1, T) if ph[i] != ph[i - 1]]
for ax in axs:
    ax.tick_params(labelsize=7)
    ax.set_xlim(t[0], t[-1])
    for b in bounds:
        ax.axvline(t[b], color="gray", ls="--", lw=0.6)
    for i in np.where(ex["done"])[0]:
        ax.axvline(t[i], color="red", ls=":", lw=0.8)
for b, name in zip([0] + bounds, [ph[0]] + [ph[i] for i in bounds]):
    axs[0].text(t[b], axs[0].get_ylim()[1], name, fontsize=6, va="bottom")
fig.canvas.draw()
panel = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
x0 = axs[3].get_position().x0 * W
x1 = axs[3].get_position().x1 * W
y0, y1 = (1 - axs[0].get_position().y1) * Hh, (1 - axs[3].get_position().y0) * Hh

cap = cv2.VideoCapture(args.raw)
tmp = args.out.replace(".mp4", "_tmp.mp4")
writer = None
for i in range(T):
    ok, frame = cap.read()
    if not ok:
        break
    p = panel.copy()
    x = int(x0 + (t[i] - t[0]) / (t[-1] - t[0]) * (x1 - x0))
    cv2.line(p, (x, int(y0)), (x, int(y1)), (0, 0, 0), 2)
    frame = cv2.resize(frame, (int(frame.shape[1] * Hh / frame.shape[0]), Hh))
    out = np.hstack([frame, p])
    if writer is None:
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (out.shape[1], out.shape[0]))
    writer.write(out)
writer.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.out], check=True)
import os

os.remove(tmp)
print("saved", args.out)
