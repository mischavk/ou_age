"""Posterior predictive check (PPC) analysis and figures for IAT.

Loads PPC metrics saved by the parameter estimation script,
computes DDM-OUM differences, and generates figures.

Each metric is a fit discrepancy (lower = closer to data); figures/prints show
DDM - OUM, so negative => DDM fits better, positive => OUM fits better.

The reported RT-quantile RMSEs cover both the congruent and incongruent
conditions (correct and error responses).

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

# As in run_analyses.py: a few hundred participants appear twice (same session_id
# prepared into two data chunks). Drop duplicate ids so the PPC differences are
# not double-counted. Both CSVs share the id ordering, so deduping each the same
# way keeps them row-aligned for the elementwise DDM - OUM differences.
oum_df = oum_df.drop_duplicates(subset="id").reset_index(drop=True)
ddm_df = ddm_df.drop_duplicates(subset="id").reset_index(drop=True)
assert (oum_df["id"].values == ddm_df["id"].values).all(), \
    "DDM/OUM PPC result CSVs are not row-aligned after dedup"

n_ppc_oum = oum_df["rms_median_c_congruent"].notna().sum()
n_ppc_ddm = ddm_df["rms_median_c_congruent"].notna().sum()
print(f"\n  OUM: {len(oum_df)} total, {n_ppc_oum} with PPC")
print(f"  DDM: {len(ddm_df)} total, {n_ppc_ddm} with PPC")

# ── RT-quantile RMSE columns, grouped by condition x outcome ──
QSUFFIX = ["median", "q1", "q3", "q7", "q9"]
RMSE_GROUPS = {
    "correct congruent":   [f"rms_{q}_c_congruent" for q in QSUFFIX],
    "error congruent":     [f"rms_{q}_e_congruent" for q in QSUFFIX],
    "correct incongruent": [f"rms_{q}_c_incongruent" for q in QSUFFIX],
    "error incongruent":   [f"rms_{q}_e_incongruent" for q in QSUFFIX],
}

print("\nRT-quantile RMSE differences (DDM - OUM, negative => DDM fits better):")
for label, cols in RMSE_GROUPS.items():
    diff = np.nanmean((ddm_df[cols] - oum_df[cols]).values)
    print(f"  {label:22s}: {diff:+.4f}")

print("\nAccuracy discrepancy differences (DDM - OUM):")
for col in ["acc_err_congruent", "acc_err_incongruent"]:
    diff = np.nanmean(ddm_df[col] - oum_df[col])
    print(f"  {col:22s}: {diff:+.4f}")

# ── Figure: PPC comparison ──
print("\nGenerating PPC figure...")

categories = ["Acc\ncong.", "Acc\ninc.",
              "RMSE\ncorrect\ncong.", "RMSE\nerror\ncong.",
              "RMSE\ncorrect\ninc.", "RMSE\nerror\ninc."]


def flat_diff(cols):
    raw = (ddm_df[cols] - oum_df[cols]).values.flatten()
    return raw[~np.isnan(raw)]


acc_diffs = [
    (ddm_df["acc_err_congruent"] - oum_df["acc_err_congruent"]).dropna().values,
    (ddm_df["acc_err_incongruent"] - oum_df["acc_err_incongruent"]).dropna().values,
]
rmse_diffs = [flat_diff(cols) for cols in RMSE_GROUPS.values()]
all_values = acc_diffs + rmse_diffs

fig, ax = plt.subplots(figsize=(11, 6))
positions = range(len(categories))

# Fixed view: the bulk of the differences (and all the central tendencies) lie
# well within +/-0.2. A handful of error-RT-RMSE differences are far larger
# (sparse error trials -> noisy quantiles) and deliberately fall off-view so the
# informative range stays legible.
YLIM = 0.2
rng = np.random.default_rng(0)

for pos, vals in zip(positions, all_values):
    if len(vals) > 5000:
        vals_plot = rng.choice(vals, 5000, replace=False)
    else:
        vals_plot = vals
    ax.scatter(
        np.full_like(vals_plot, pos) + rng.uniform(-0.15, 0.15, len(vals_plot)),
        vals_plot, alpha=0.05, s=5, color="gray",
    )
    med = np.nanmedian(vals)
    med_color = "tab:blue" if med >= 0 else "tab:orange"
    ax.plot([pos - 0.3, pos + 0.3], [med] * 2,
            color=med_color, linewidth=2.5, zorder=5)
    mean_val = np.nanmean(vals)
    mean_color = "tab:blue" if mean_val >= 0 else "tab:orange"
    ax.plot([pos - 0.3, pos + 0.3], [mean_val] * 2,
            color=mean_color, linewidth=2, linestyle="--", zorder=5)

_all_finite = np.concatenate(all_values)
_n_clipped = int(np.sum(np.abs(_all_finite) > YLIM))
ax.set_ylim(-YLIM, YLIM)
print(f"  y-limit fixed to +/-{YLIM} "
      f"({_n_clipped}/{_all_finite.size} points beyond view)")

ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_ylim(-0.25, 0.25)
ax.set_xticks(positions)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("DDM − OUM discrepancy (negative ⇒ DDM fits better)", fontsize=12)
ax.set_title("IAT — PPC metric differences", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

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
