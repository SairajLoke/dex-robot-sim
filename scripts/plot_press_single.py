"""Plot the single-fingertip press results: rigid-sphere press (GT force, force_torque, elastomer, probe distance) and the ghost-sphere penetration test (elastomer vs probe penetration depth)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "docs/data"
out = sys.argv[2] if len(sys.argv) > 2 else "docs/data/press_single_summary.png"


def load(n):
    z = np.load(f"{d}/{n}.npz")
    return {c: z["data"][:, i] for i, c in enumerate(z["cols"])}


solid, ghost, fine = load("press_single_deep"), load("press_single_ghost"), load("press_single_ghost_fine")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(solid["step"], solid["gt_force"], "C3", label="GT contact force (N)")
ax[0].set_ylim(0, 12)
ax[0].set_xlabel("step (the 76 N one-step impact spike at step ~116 is clipped)")
ax[0].set_ylabel("N")
ax2 = ax[0].twinx()
ax2.plot(solid["step"], solid["ft_sum"], "C0", label="force_torque taxel sum")
ax2.plot(solid["step"], solid["el_max"], "C1", lw=3, label="elastomer max |disp|")
ax2.set_ylim(0, 1.2)
ax[0].set_title(f"Rigid sphere press: elastomer stays 0\n(closest probe {solid['probe_min_sd'].min()*1e3:.2f} mm OUTSIDE surface)")
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, loc="center right")
ax[1].plot(solid["step"], solid["probe_min_sd"] * 1e3, label="rigid sphere")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylim(-5, 20)
ax[1].set_xlabel("step")
ax[1].set_ylabel("min probe signed distance to sphere (mm); <0 = inside")
ax[1].set_title("A probe never gets inside a rigid object")
ax[1].legend()
for g, lab, c in ((ghost, "ghost run 1 (bend 0.3-1.6)", "C0"), (fine, "ghost run 2 (bend 0.5-0.62)", "C1")):
    pen = -g["probe_min_sd"] * 1e3
    ax[2].scatter(pen, g["el_max"], s=7, c=c, label=lab)
ax[2].set_xlim(-5, 40)
ax[2].set_xlabel("max probe penetration depth into sphere (mm)")
ax[2].set_ylabel("elastomer max |displacement|")
ax[2].set_title("Ghost sphere: elastomer vs probe penetration depth")
ax[2].legend()
plt.tight_layout()
plt.savefig(out, dpi=110)
print("saved", out)
