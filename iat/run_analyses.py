"""Age trend analyses for IAT data.

Loads DDM and OUM parameter estimates, computes age trends for all parameters,
and generates demographic subgroup analyses for the self-excitation parameter k.

Figures:
  - Figure 5: 3×2 grid of DDM vs OUM age trajectories (v, a, ndt)
  - Figure 6: 2×2 grid of k age trajectories (overall + demographic splits)

Usage:
    python run_analyses.py
"""

import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = "iat_data/"

# ── Load data ──
print("Loading parameter estimates...")
oum_df = pd.read_csv(os.path.join(DATA_DIR, "estimates", "iat_results_oum.csv"))
ddm_df = pd.read_csv(os.path.join(DATA_DIR, "estimates", "iat_results_ddm.csv"))

joint_df = oum_df.join(ddm_df, lsuffix="_oum", rsuffix="_ddm")
# The join is positional; guard against mismatched estimation runs.
assert len(oum_df) == len(ddm_df) and (
    joint_df["id_oum"].values == joint_df["id_ddm"].values
).all(), "DDM/OUM result CSVs are not row-aligned (different estimation runs?)"
# A small number of participants (~939) appear twice in the estimates because the
# same session_id was prepared into two data chunks. Keep one row per participant
# so every downstream analysis (age trends, correlations, and the subgroup plots)
# counts each person once. This is separate from the demographics-file dedup
# below, which guards the merge against duplicate session_ids on that side.
joint_df = joint_df.drop_duplicates(subset="id_ddm")
joint_df = joint_df[(joint_df["age_ddm"] >= 10) & (joint_df["age_ddm"] <= 80)]

print(f"  OUM: {len(oum_df)} persons")
print(f"  DDM: {len(ddm_df)} persons")
print(f"  Joint (age 10-80): {len(joint_df)} persons")

# ── Figure 5: Age trends for DDM/OUM parameters (3×2) ──
print("\nGenerating Figure 5 (parameter age trends)...")

param_configs = [
    # Row 1: Drift rates
    ("v1", "Drift rate (congruent)", (0, 3)),
    ("v2", "Drift rate (incongruent)", (0, 3)),
    # Row 2: Boundary separations
    ("a1", "Boundary separation (congruent)", (0, 6)),
    ("a2", "Boundary separation (incongruent)", (0, 6)),
    # Row 3: Non-decision times
    ("ndt_correct", "Non-decision time (correct)", (0, 0.7)),
    ("ndt_error", "Non-decision time (error)", (0, 1.7)),
]

panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes = axes.flatten()

for idx, (param, title, ylim) in enumerate(param_configs):
    ax = axes[idx]

    sns.lineplot(
        data=joint_df.sort_values("age_ddm"),
        x="age_ddm", y=f"{param}_ddm",
        estimator=np.mean, errorbar="sd",
        marker="o", label="DDM", ax=ax,
    )
    sns.lineplot(
        data=joint_df.sort_values("age_ddm"),
        x="age_ddm", y=f"{param}_oum",
        estimator=np.mean, errorbar="sd",
        marker="o", label="OUM", ax=ax,
    )
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel(param)
    ax.legend(title="Model", fontsize=9)
    ax.text(-0.08, 1.05, panel_labels[idx], transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}figure5_age_trends_iat.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved {FIGURES_DIR}figure5_age_trends_iat.pdf")

# ── Age correlations (Table 3) ──
print("\nAge correlations:")
print("-" * 50)

corr_rows = []
for param in ["v1", "v2", "a1", "a2", "ndt_correct", "ndt_error"]:
    r_ddm = joint_df["age_ddm"].corr(joint_df[f"{param}_ddm"])
    r_oum = joint_df["age_oum"].corr(joint_df[f"{param}_oum"])
    corr_rows.append((param, r_ddm, r_oum))
    print(f"  {param:15s}  DDM r={r_ddm:+.3f}  OUM r={r_oum:+.3f}")
r_k = joint_df["age_oum"].corr(joint_df["k"])
print(f"  {'k':15s}  OUM r={r_k:+.3f}")

# Write LaTeX table
param_labels = {
    "v1": r"Drift rate (congruent, $v_1$)",
    "v2": r"Drift rate (incongruent, $v_2$)",
    "a1": r"Boundary separation (congruent, $a_1$)",
    "a2": r"Boundary separation (incongruent, $a_2$)",
    "ndt_correct": r"Non-decision time (correct, $\tau_c$)",
    "ndt_error": r"Non-decision time (error, $\tau_e$)",
}

tex_path = "table3_age_correlations_iat.tex"
with open(tex_path, "w") as f:
    f.write(r"""\begin{table}[ht]
\centering
\caption{Pearson age correlations for DDM and OUM parameters in the IAT (Study 2, $N = """ + f"{len(joint_df):,}" + r"""$).}
\label{tab:age_correlations}
\begin{tabular}{l cc}
\toprule
Parameter & DDM & OUM \\
\midrule
""")
    for param, r_ddm, r_oum in corr_rows:
        f.write(f"{param_labels[param]} & ${r_ddm:+.3f}$ & ${r_oum:+.3f}$ "
                r"\\" + "\n")
    f.write(r"\addlinespace" + "\n")
    f.write(r"Self-excitation ($k$) & --- & $" + f"{r_k:+.3f}" + r"$ \\" + "\n")
    f.write(r"""\bottomrule
\end{tabular}

\smallskip
\noindent\textit{Note.} Correlations are computed from the median of the posterior parameter estimates per participant, restricted to ages 10--80.
\end{table}
""")
print(f"  Saved {tex_path}")

# ── Figure 6: k age trajectories (2×2) ──
demo_path = os.path.join(DATA_DIR, "estimates", "iat_demographics.csv")
if os.path.exists(demo_path):
    print("\nLoading demographics...")
    # Drop duplicate session_ids in demographics_df so the merge does not
    # double-count participants into the subgroup statistics.
    demographics_df = pd.read_csv(demo_path, low_memory=False).drop_duplicates(
        subset="session_id"
    )
    joint_df_demo = joint_df.merge(
        demographics_df, left_on="id_ddm", right_on="session_id"
    )
    custom_palette = sns.color_palette(["#249424", "#C42003"])

    print("Generating Figure 6 (k age trajectories)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # a) Overall k
    ax = axes[0, 0]
    sns.lineplot(
        data=joint_df.sort_values("age_ddm"),
        x="age_ddm", y="k",
        estimator=np.mean, errorbar="sd",
        marker="o", color="tab:orange", ax=ax,
    )
    ax.set_ylim(0, 4.2)
    ax.set_title("Self-excitation (k)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Self-excitation (k)")
    ax.text(-0.08, 1.05, "a)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    # b) By gender
    ax = axes[0, 1]
    # Project Implicit coding:
    # birthsex 1 = Male, 2 = Female.
    df_gender = joint_df_demo.copy()
    df_gender = df_gender[(df_gender["birthsex"] == 1) | (df_gender["birthsex"] == 2)]
    df_gender["birthsex"] = df_gender["birthsex"].map({1: "Male", 2: "Female"})
    sns.lineplot(
        data=df_gender.sort_values("age_ddm"),
        x="age_ddm", y="k",
        estimator=np.mean, errorbar="sd",
        marker="o", hue="birthsex",
        palette=custom_palette, ax=ax,
    )
    ax.set_ylim(0, 4.2)
    ax.set_title("k by gender", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Self-excitation (k)")
    ax.legend(title="Gender")
    ax.text(-0.08, 1.05, "b)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    # c) By location
    # Residence coding changed across collection years: 2003-2014 store "US" /
    # country codes / "other country" in `is_us`; 2015-2016 store countryres
    # codes there ("1" = US); 2017+ leave `is_us` empty and use `countryres`
    # ("1" = US). A participant counts as US-resident if either column says US;
    # rows where both columns are missing are excluded (not counted as non-US).
    ax = axes[1, 0]
    us_codes = {"US", "1", "1.0"}
    missing_codes = {"nan", "<NA>", "None", "", ".", "-999", "-999.0"}
    isu = joint_df_demo["is_us"].astype(str).str.strip()
    cres = joint_df_demo["countryres"].astype(str).str.strip()
    known = ~isu.isin(missing_codes) | ~cres.isin(missing_codes)
    joint_df_demo["is_us"] = isu.isin(us_codes) | cres.isin(us_codes)
    df_loc = joint_df_demo[known]
    sns.lineplot(
        data=df_loc.sort_values("age_ddm"),
        x="age_ddm", y="k",
        estimator=np.mean, errorbar="sd",
        marker="o", hue="is_us",
        palette=custom_palette, ax=ax,
    )
    ax.set_ylim(0, 4.2)
    ax.set_title("k by location", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Self-excitation (k)")
    ax.legend(title="From US")
    ax.text(-0.08, 1.05, "c)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    # d) By education
    ax = axes[1, 1]
    joint_df_demo["college_edu"] = joint_df_demo["edu"] >= 5
    df_college = joint_df_demo[joint_df_demo["age_ddm"] >= 17].copy()
    sns.lineplot(
        data=df_college.sort_values("age_ddm"),
        x="age_ddm", y="k",
        estimator=np.mean, errorbar="sd",
        marker="o", hue="college_edu",
        palette=custom_palette, ax=ax,
    )
    ax.set_ylim(0, 4.2)
    ax.set_title("k by education", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Self-excitation (k)")
    ax.legend(title="College education")
    ax.text(-0.08, 1.05, "d)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}figure6_k_subgroups_iat.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR}figure6_k_subgroups_iat.pdf")
else:
    print(f"\n  Demographics file not found at {demo_path}, skipping subgroup analyses.")

print("\nDone.")
