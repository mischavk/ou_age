"""Generate Figure 1 — DDM vs. OUM evidence accumulation schematic.

Three panels:
  (a) DDM: Linear drift paths with Gaussian noise
  (b) OUM: Self-excitatory drift paths (k > 0)
  (c) Comparison of resulting RT distributions

Usage:
    python run_figure1_schematic.py
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from sfi_functions import sfi_simulator_fun

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {"DDM": "tab:blue", "OUM": "tab:orange"}


def simulate_paths(v, a, k=0.0, n_paths=10, dt=0.001, max_time=1.0, seed=42):
    """Simulate evidence accumulation paths for schematic illustration."""
    max_steps = int(max_time / dt)
    rng = np.random.default_rng(seed)
    paths = []
    attempts = 0
    while len(paths) < n_paths and attempts < 500:
        attempts += 1
        x, t = 0.0, 0.0
        xs, ts = [x], [t]
        hit = False
        for step in range(max_steps):
            x += (v + k * x) * dt + np.sqrt(dt) * rng.normal()
            t += dt
            xs.append(x)
            ts.append(t)
            if x >= a / 2 or x <= -a / 2:
                hit = True
                break
        if hit:
            paths.append((np.array(ts), np.array(xs)))
    return paths


def plot_paths(ax, paths, a, color, alpha=0.6):
    """Draw accumulation paths and boundary lines."""
    ax.axhline(a / 2, color="black", linewidth=2)
    ax.axhline(-a / 2, color="black", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for ts, xs in paths:
        ax.plot(ts, xs, color=color, alpha=alpha, linewidth=1)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Evidence", fontsize=13)
    ax.set_ylim(-a / 2 - 0.3, a / 2 + 0.3)
    ax.set_xlim(0, 1.0)


# Simulation parameters
V, A_DDM, A_OUM, K = 1.5, 2.5, 2.5, 5.0
N_PATHS = 10

print("Simulating DDM and OUM paths...")
ddm_paths = simulate_paths(v=V, a=A_DDM, k=0.0, n_paths=N_PATHS, seed=0)
oum_paths = simulate_paths(v=V, a=A_OUM, k=K, n_paths=N_PATHS, seed=0)

# RT distributions via full simulator
N_TRIALS = 2000
ddm_rts = np.concatenate([
    sfi_simulator_fun(V, A_DDM, ndt=0.0, k=0.0)
    for _ in range(N_TRIALS // 100)
])
oum_rts = np.concatenate([
    sfi_simulator_fun(V, A_OUM, ndt=0.0, k=K)
    for _ in range(N_TRIALS // 100)
])
ddm_correct = ddm_rts[ddm_rts > 0]
oum_correct = oum_rts[oum_rts > 0]

print(f"DDM median correct RT: {np.median(ddm_correct):.3f} s")
print(f"OUM median correct RT: {np.median(oum_correct):.3f} s")

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): DDM
plot_paths(axs[0], ddm_paths, A_DDM, color=COLORS["DDM"])
axs[0].set_title("(a) DDM", fontsize=15, fontweight="bold")
axs[0].text(
    0.97, 0.97, r"$dx = v \cdot dt + dW$",
    transform=axs[0].transAxes, ha="right", va="top",
    fontsize=11, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
axs[0].text(0.02, A_DDM / 2 + 0.05, "Upper boundary", fontsize=9, va="bottom")
axs[0].text(0.02, -A_DDM / 2 - 0.05, "Lower boundary", fontsize=9, va="top")

# Panel (b): OUM
plot_paths(axs[1], oum_paths, A_OUM, color=COLORS["OUM"])
axs[1].set_title("(b) OUM", fontsize=15, fontweight="bold")
axs[1].text(
    0.97, 0.97, r"$dx = (v + k \cdot x) \cdot dt + dW$",
    transform=axs[1].transAxes, ha="right", va="top",
    fontsize=11, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
axs[1].set_ylabel("")

# Panel (c): RT distributions
bins = np.linspace(0, 1.0, 50)
axs[2].hist(ddm_correct, bins=bins, density=True, alpha=0.6,
            color=COLORS["DDM"], label="DDM")
axs[2].hist(oum_correct, bins=bins, density=True, alpha=0.6,
            color=COLORS["OUM"], label="OUM")
axs[2].set_xlabel("Response time (s)", fontsize=13)
axs[2].set_ylabel("Density", fontsize=13)
axs[2].set_title("(c) RT distributions", fontsize=15, fontweight="bold")
axs[2].legend(fontsize=12, frameon=False)
axs[2].set_xlim(0, 1.0)

plt.tight_layout()
save_path = f"{FIGURES_DIR}figure1_schematic.pdf"
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {save_path}")
