"""Pool elastomer recordings and bin fingertip displacement against ground-truth contact force and distance.

    python scripts/plot_elastomer_dose_response.py docs/data/press_el docs/data/rec_el docs/data/rec_el_r3
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIPS = ["index", "middle", "ring", "thumb"]
mx, nl, gt, sd = [], [], [], []
for stem in sys.argv[1:]:
    d, e = np.load(stem + ".npz", allow_pickle=True), np.load(stem + "_extra.npz")
    T = len(e["t"])
    for tip in TIPS:
        a = d[f"tactile_elastomer_{tip}_3_tip__data"]
        a = np.median(a.reshape(T, 5, a.shape[1] // 5, 3), axis=1)
        mag = np.linalg.norm(a, axis=-1)
        mx.append(mag.max(1) * 1e3)
        nl.append((mag > 1e-6).sum(1))
        gt.append(np.linalg.norm(e[f"priv_contact_{tip}_3_tip"], axis=-1))
        sd.append(e[f"priv_surface_distance_{tip}_3_tip"].reshape(T, -1)[:, 0] * 1e3)
mx, nl, gt, sd = map(np.concatenate, (mx, nl, gt, sd))
print(f"n={len(mx)}  corr(max disp, GT)={np.corrcoef(mx, gt)[0, 1]:.2f}  corr(n loaded, GT)={np.corrcoef(nl, gt)[0, 1]:.2f}")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
edges = [0, 0.1, 1, 2, 4, 100]
lab = ["<0.1", "0.1-1", "1-2", "2-4", ">4"]
idx = np.digitize(gt, edges[1:-1])
ax[0].bar(lab, [np.mean(nl[idx == i] > 0) for i in range(5)])
ax[0].set(xlabel="GT hand-object |F| (N)", ylabel="fraction of steps with any taxel displaced", title="elastomer loaded vs GT force")
ax[1].bar(lab, [mx[idx == i].mean() for i in range(5)])
ax[1].set(xlabel="GT hand-object |F| (N)", ylabel="mean max taxel displacement (mm)", title="elastomer displacement vs GT force")
for i in range(5):
    ax[0].text(i, 0.01, f"n={(idx == i).sum()}", ha="center", fontsize=7)
de = [0, 2, 5, 10, 20, 1e9]
dl = ["<2", "2-5", "5-10", "10-20", ">20"]
di = np.digitize(sd, de[1:-1])
ax[2].bar(dl, [np.mean(nl[di == i] > 0) for i in range(5)])
ax[2].set(xlabel="fingertip-object surface distance (mm)", ylabel="fraction loaded", title="elastomer loaded vs distance")
fig.tight_layout()
fig.savefig("docs/data/elastomer_dose_response.png", dpi=130)
