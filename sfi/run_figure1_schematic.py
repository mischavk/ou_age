"""Generate Figure 1 — DDM vs. OUM evidence accumulation schematic.

Three panels:
  (a) DDM: Linear drift paths with Gaussian noise
  (b) OUM: Self-excitatory drift paths (k > 0)
  (c) Comparison of resulting RT distributions

The path/RT parameters are NOT illustrative defaults — they are the posterior
median estimates of a single real FF1 participant (chosen for a high OUM
self-excitation, k > 5). The DDM panel uses that participant's DDM posterior
medians and the OUM panel uses their OUM posterior medians, so the figure shows
how the two models account for the SAME empirical data with different latent
dynamics (DDM: narrow boundary + linear drift; OUM: wide boundary + self-
excitation).

Usage:
    python run_figure1_schematic.py
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sfi_functions import sfi_simulator_fun

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Participant whose posterior medians parameterise the schematic. FF1, OUM k>5.
TASK = "FF1"
PERSON_IDX = 6

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


def _posterior_medians(model):
    """Posterior median (v, a, ndt[, k]) for PERSON_IDX in TASK."""
    d = np.load(f"sfi_data/{TASK}_sfi_{'fast'}_{model}_estimates.npy",
                allow_pickle=True).item()
    med = {}
    for key in d:
        arr = np.asarray(d[key])           # (n_persons, n_samples, 1)
        med[key] = float(np.median(arr[PERSON_IDX]))
    return med


def simulate_paths(v, a, k=0.0, n_paths=10, dt=0.001, max_time=2.0, seed=0):
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


def plot_paths(ax, paths, a, color, xmax, alpha=0.6):
    """Draw accumulation paths and boundary lines."""
    ax.axhline(a / 2, color="black", linewidth=2)
    ax.axhline(-a / 2, color="black", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for ts, xs in paths:
        ax.plot(ts, xs, color=color, alpha=alpha, linewidth=1)
    ax.set_xlabel("Decision time (s)", fontsize=13)
    ax.set_ylabel("Evidence", fontsize=13)
    ax.set_ylim(-a / 2 - 0.3, a / 2 + 0.3)
    ax.set_xlim(0, xmax)


# ── Participant-specific parameters (posterior medians) ──
ddm = _posterior_medians("ddm")
oum = _posterior_medians("oum")
V_DDM, A_DDM, NDT_DDM = ddm["v"], ddm["a"], ddm["ndt"]
V_OUM, A_OUM, NDT_OUM, K = oum["v"], oum["a"], oum["ndt"], oum["k"]
N_PATHS = 10

print(f"{TASK} participant #{PERSON_IDX} posterior medians:")
print(f"  DDM: v={V_DDM:.2f} a={A_DDM:.2f} ndt={NDT_DDM:.2f}")
print(f"  OUM: v={V_OUM:.2f} a={A_OUM:.2f} ndt={NDT_OUM:.2f} k={K:.2f}")

print("Simulating DDM and OUM paths...")
ddm_paths = simulate_paths(v=V_DDM, a=A_DDM, k=0.0, n_paths=N_PATHS, seed=13)
oum_paths = simulate_paths(v=V_OUM, a=A_OUM, k=K, n_paths=N_PATHS, seed=13)

# Shared decision-time axis for the two path panels.
path_xmax = 1.05 * max(
    max(ts[-1] for ts, _ in ddm_paths),
    max(ts[-1] for ts, _ in oum_paths),
)

# RT distributions via full simulator (include the participant's ndt).
N_TRIALS = 4000
ddm_rts = np.concatenate([
    sfi_simulator_fun(V_DDM, A_DDM, ndt=NDT_DDM, k=0.0)
    for _ in range(N_TRIALS // 100)
])
oum_rts = np.concatenate([
    sfi_simulator_fun(V_OUM, A_OUM, ndt=NDT_OUM, k=K)
    for _ in range(N_TRIALS // 100)
])
ddm_correct = ddm_rts[ddm_rts > 0]
oum_correct = oum_rts[oum_rts > 0]

print(f"DDM median correct RT: {np.median(ddm_correct):.3f} s")
print(f"OUM median correct RT: {np.median(oum_correct):.3f} s")

rt_xmax = 1.05 * np.percentile(np.concatenate([ddm_correct, oum_correct]), 99)

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Panel (a): DDM
plot_paths(axs[0], ddm_paths, A_DDM, color=COLORS["DDM"], xmax=path_xmax)
axs[0].set_title("(a) DDM", fontsize=15, fontweight="bold")
axs[0].text(
    0.97, 0.97, r"$dx = v \cdot dt + dW$" + f"\n$v$={V_DDM:.2f}, $a$={A_DDM:.2f}",
    transform=axs[0].transAxes, ha="right", va="top",
    fontsize=11, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
axs[0].text(0.02, A_DDM / 2 + 0.05, "Upper boundary", fontsize=9, va="bottom")
axs[0].text(0.02, -A_DDM / 2 - 0.05, "Lower boundary", fontsize=9, va="top")

# Panel (b): OUM
plot_paths(axs[1], oum_paths, A_OUM, color=COLORS["OUM"], xmax=path_xmax)
axs[1].set_title("(b) OUM", fontsize=15, fontweight="bold")
axs[1].text(
    0.97, 0.97,
    r"$dx = (v + k \cdot x) \cdot dt + dW$" + f"\n$v$={V_OUM:.2f}, $a$={A_OUM:.2f}, $k$={K:.2f}",
    transform=axs[1].transAxes, ha="right", va="top",
    fontsize=11, style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
axs[1].set_ylabel("")

# Panel (c): RT distributions as kernel-density lines (less overlap than the
# overlapping semi-transparent histograms).
rt_grid = np.linspace(0, rt_xmax, 400)
for rts, model in [(ddm_correct, "DDM"), (oum_correct, "OUM")]:
    density = gaussian_kde(rts)(rt_grid)
    axs[2].plot(rt_grid, density, color=COLORS[model], linewidth=2.2, label=model)
    axs[2].fill_between(rt_grid, density, color=COLORS[model], alpha=0.12)
axs[2].set_ylim(bottom=0)
axs[2].set_xlabel("Response time (s)", fontsize=13)
axs[2].set_ylabel("Density", fontsize=13)
axs[2].set_title("(c) RT distributions", fontsize=15, fontweight="bold")
axs[2].legend(fontsize=12, frameon=False)
axs[2].set_xlim(0, rt_xmax)

fig.suptitle(
    f"DDM vs. OUM fitted to the same participant "
    f"({TASK} #{PERSON_IDX}; posterior medians, OUM $k$={K:.1f})",
    fontsize=13, y=1.02,
)

plt.tight_layout()
save_path = f"{FIGURES_DIR}figure1_schematic.pdf"
fig.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {save_path}")
