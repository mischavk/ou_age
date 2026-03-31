"""Posterior predictive check (PPC) analysis and figures for IAT.

Loads PPC metrics saved by the parameter estimation script,
computes DDM-OUM differences, and generates figures.

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

DATA_DIR = "iat_data/"

# ── Load data ──
print("=" * 60)
print("Posterior Predictive Checks — DDM vs. OUM (IAT)")
print("=" * 60)

oum_df = pd.read_csv(os.path.join(DATA_DIR, "estimates", "iat_results_oum.csv"))
ddm_df = pd.read_csv(os.path.join(DATA_DIR, "estimates", "iat_results_ddm.csv"))

n_ppc_oum = oum_df["wd_c_congruent"].notna().sum()
n_ppc_ddm = ddm_df["wd_c_congruent"].notna().sum()
print(f"\n  OUM: {len(oum_df)} total, {n_ppc_oum} with PPC")
print(f"  DDM: {len(ddm_df)} total, {n_ppc_ddm} with PPC")

# ── Wasserstein distance differences ──
# OUM - DDM so positive = DDM has lower error = DDM fits better
print("\nWasserstein distance (OUM - DDM, positive = DDM better):")
wd_cols = ["wd_c_congruent", "wd_e_congruent", "wd_c_incongruent", "wd_e_incongruent"]
for col in wd_cols:
    diff = np.nanmean(oum_df[col] - ddm_df[col])
    print(f"  {col:25s}: {diff:+.4f}")

# ── RMSE differences ──
rmse_cols = [
    "rms_median_c_congruent", "rms_q1_c_congruent", "rms_q3_c_congruent",
    "rms_q7_c_congruent", "rms_q9_c_congruent",
    "rms_median_e_congruent", "rms_q1_e_congruent", "rms_q3_e_congruent",
    "rms_q7_e_congruent", "rms_q9_e_congruent",
]

print("\nRMSE differences (OUM - DDM, positive = DDM better):")
print("  Correct-congruent:")
for col in rmse_cols[:5]:
    diff = np.nanmean(oum_df[col] - ddm_df[col])
    print(f"    {col:35s}: {diff:+.4f}")

print("  Error-congruent:")
for col in rmse_cols[5:]:
    diff = np.nanmean(oum_df[col] - ddm_df[col])
    print(f"    {col:35s}: {diff:+.4f}")

# ── Figure: PPC comparison ──
print("\nGenerating PPC figures...")

# Compute mean differences per metric category
categories = ["RMSE\ncorrect\ncong.", "RMSE\nerror\ncong.",
              "WD\ncorrect\ncong.", "WD\nerror\ncong.",
              "WD\ncorrect\ninc.", "WD\nerror\ninc."]

# OUM - DDM so positive = DDM has lower error = DDM fits better
rmse_c_diffs_raw = (oum_df[rmse_cols[:5]] - ddm_df[rmse_cols[:5]]).values.flatten()
rmse_c_diffs = rmse_c_diffs_raw[~np.isnan(rmse_c_diffs_raw)]
rmse_e_diffs_raw = (oum_df[rmse_cols[5:]] - ddm_df[rmse_cols[5:]]).values.flatten()
rmse_e_diffs = rmse_e_diffs_raw[~np.isnan(rmse_e_diffs_raw)]
wd_diffs = [
    (oum_df["wd_c_congruent"] - ddm_df["wd_c_congruent"]).dropna().values,
    (oum_df["wd_e_congruent"] - ddm_df["wd_e_congruent"]).dropna().values,
    (oum_df["wd_c_incongruent"] - ddm_df["wd_c_incongruent"]).dropna().values,
    (oum_df["wd_e_incongruent"] - ddm_df["wd_e_incongruent"]).dropna().values,
]

fig, ax = plt.subplots(figsize=(10, 6))

all_values = [rmse_c_diffs, rmse_e_diffs] + wd_diffs
positions = range(len(categories))

for pos, vals in zip(positions, all_values):
    # Subsample for plotting if very large
    if len(vals) > 5000:
        vals_plot = np.random.choice(vals[~np.isnan(vals)], 5000, replace=False)
    else:
        vals_plot = vals[~np.isnan(vals)]

    colors = ["tab:blue" if v >= 0 else "tab:orange" for v in vals_plot]
    ax.scatter(
        np.full_like(vals_plot, pos) + np.random.uniform(-0.15, 0.15, len(vals_plot)),
        vals_plot, alpha=0.05, s=5, c=colors,
    )
    med = np.nanmedian(vals)
    med_color = "tab:blue" if med >= 0 else "tab:orange"
    ax.plot([pos - 0.3, pos + 0.3], [med] * 2,
            color=med_color, linewidth=2.5, zorder=5)
    mean_val = np.nanmean(vals)
    mean_color = "tab:blue" if mean_val >= 0 else "tab:orange"
    ax.plot([pos - 0.3, pos + 0.3], [mean_val] * 2,
            color=mean_color, linewidth=2, linestyle="--", zorder=5)

ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_ylim(-0.25, 0.25)
ax.set_xticks(positions)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("OUM − DDM (blue = DDM better, orange = OUM better)", fontsize=12)
ax.set_title("IAT — PPC metric differences", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2.5, label="Median"),
    Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="Mean"),
]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}figureC5_ppc_comparison_iat.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved {FIGURES_DIR}figureC5_ppc_comparison_iat.pdf")

print("\nDone.")
