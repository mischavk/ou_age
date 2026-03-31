"""Posterior predictive check (PPC) analysis and figures.

Loads PPC metrics (RMSE, Wasserstein distance) saved by the parameter
estimation scripts, computes DDM-OUM differences, and generates figures.

Positive values = DDM fits better, negative = OUM fits better.

Usage:
    python run_ppc_analysis.py
"""

import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

METRIC_COLS = [
    "WD correct", "WD error",
    "RMSE Q1 C", "RMSE Q3 C", "RMSE Q5 C", "RMSE Q7 C", "RMSE Q9 C",
    "RMSE Q1 E", "RMSE Q3 E", "RMSE Q5 E", "RMSE Q7 E", "RMSE Q9 E",
]


def compute_ppc_metrics(prefix: str, speed: str) -> pd.DataFrame:
    """Compute DDM-OUM PPC metric differences for 9 tasks."""
    files = sorted([
        os.path.join("sfi_data", f)
        for f in os.listdir("sfi_data")
        if f.endswith(".txt") and f.startswith(prefix)
    ])

    metrics = np.zeros((len(files), 12))
    task_names = []

    for i, file in enumerate(files):
        task_names.append(os.path.basename(file).replace(".txt", ""))

        ppc_ddm = np.load(
            file.replace(".txt", f"_sfi_{speed}_ddm_ppc.npy"),
            allow_pickle=True,
        )[()]
        ppc_oum = np.load(
            file.replace(".txt", f"_sfi_{speed}_oum_ppc.npy"),
            allow_pickle=True,
        )[()]

        # Wasserstein distance (correct, error)
        for col in range(2):
            metrics[i, col] = (
                np.nanmean(ppc_ddm["wd"][:, col])
                - np.nanmean(ppc_oum["wd"][:, col])
            )

        # RMSE at 5 quantiles x 2 (correct, error)
        for col in range(10):
            metrics[i, col + 2] = (
                np.nanmean(ppc_ddm["rmse"][:, col])
                - np.nanmean(ppc_oum["rmse"][:, col])
            )

    return pd.DataFrame(metrics, index=task_names, columns=METRIC_COLS)


def plot_ppc_comparison(df_fast: pd.DataFrame, df_slow: pd.DataFrame,
                        save_path: str) -> None:
    """Create a 2-panel figure comparing DDM-OUM PPC metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, df, title in zip(axes, [df_fast, df_slow], ["Fast tasks", "Slow tasks"]):
        # Separate RMSE correct, RMSE error, WD
        rmse_c = df[["RMSE Q1 C", "RMSE Q3 C", "RMSE Q5 C", "RMSE Q7 C", "RMSE Q9 C"]]
        rmse_e = df[["RMSE Q1 E", "RMSE Q3 E", "RMSE Q5 E", "RMSE Q7 E", "RMSE Q9 E"]]
        wd = df[["WD correct", "WD error"]]

        # Plot per-task means as points, with median line
        categories = ["RMSE\ncorrect", "RMSE\nerror", "WD\ncorrect", "WD\nerror"]
        values = [
            rmse_c.values.flatten(),
            rmse_e.values.flatten(),
            wd["WD correct"].values,
            wd["WD error"].values,
        ]

        positions = range(len(categories))
        for pos, vals in zip(positions, values):
            ax.scatter(
                np.full_like(vals, pos) + np.random.uniform(-0.1, 0.1, len(vals)),
                vals, alpha=0.4, s=20, color="gray",
            )
            ax.plot([pos - 0.25, pos + 0.25], [np.median(vals)] * 2,
                    color="red", linewidth=2.5, zorder=5)

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("DDM − OUM (positive = DDM better)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_ppc_heatmap(df: pd.DataFrame, title: str, save_path: str) -> None:
    """Create a heatmap of per-task DDM-OUM PPC differences."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(df.values, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_title(f"{title}\n(DDM − OUM: red = DDM better, blue = OUM better)",
                 fontsize=13)
    plt.colorbar(im, ax=ax, label="DDM − OUM", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ──
print("=" * 60)
print("Posterior Predictive Checks — DDM vs. OUM")
print("=" * 60)

# Fast tasks
print("\nFast tasks:")
df_fast = compute_ppc_metrics("F", "fast")
print(df_fast.median().to_string())
rmse_c_med = df_fast[["RMSE Q1 C", "RMSE Q3 C", "RMSE Q5 C", "RMSE Q7 C", "RMSE Q9 C"]].stack().median()
rmse_e_med = df_fast[["RMSE Q1 E", "RMSE Q3 E", "RMSE Q5 E", "RMSE Q7 E", "RMSE Q9 E"]].stack().median()
print(f"\n  Overall RMSE correct median: {rmse_c_med:.4f}")
print(f"  Overall RMSE error median:   {rmse_e_med:.4f}")

# Slow tasks
print("\nSlow tasks:")
df_slow = compute_ppc_metrics("S", "slow")
print(df_slow.median().to_string())
rmse_c_med = df_slow[["RMSE Q1 C", "RMSE Q3 C", "RMSE Q5 C", "RMSE Q7 C", "RMSE Q9 C"]].stack().median()
rmse_e_med = df_slow[["RMSE Q1 E", "RMSE Q3 E", "RMSE Q5 E", "RMSE Q7 E", "RMSE Q9 E"]].stack().median()
print(f"\n  Overall RMSE correct median: {rmse_c_med:.4f}")
print(f"  Overall RMSE error median:   {rmse_e_med:.4f}")

# Figures
print("\nGenerating PPC figures...")

plot_ppc_comparison(df_fast, df_slow,
                    f"{FIGURES_DIR}figureA5_ppc_comparison.pdf")
print(f"  Saved {FIGURES_DIR}figureA5_ppc_comparison.pdf")

plot_ppc_heatmap(df_fast, "Fast tasks — PPC metric differences",
                 f"{FIGURES_DIR}figureA5_ppc_heatmap_fast.pdf")
print(f"  Saved {FIGURES_DIR}figureA5_ppc_heatmap_fast.pdf")

plot_ppc_heatmap(df_slow, "Slow tasks — PPC metric differences",
                 f"{FIGURES_DIR}figureA5_ppc_heatmap_slow.pdf")
print(f"  Saved {FIGURES_DIR}figureA5_ppc_heatmap_slow.pdf")

print("\nDone.")
