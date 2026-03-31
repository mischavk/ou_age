"""Run age–parameter correlation analyses for Study 1.

Loads posterior estimates from parameter estimation scripts and computes
posterior distributions of Pearson correlations with age. Saves Figure 3
and Table 3 (LaTeX).

Usage:
    python run_analyses.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfi_functions import prepare_correlation_posteriors, plot_comparative_posteriors

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

N_SAMPLES = 3000  # matching fast estimation settings

# ── Fast tasks ──
print("Computing age correlations for fast tasks...")
cor_dict_fast, stats_fast = prepare_correlation_posteriors(slow_tasks=False, n_samples=N_SAMPLES)
fig = plot_comparative_posteriors(
    cor_dict_fast,
    save_path=f"{FIGURES_DIR}figure3a_age_correlations_fast.pdf",
)
plt.close(fig)
print(f"  Saved {FIGURES_DIR}figure3a_age_correlations_fast.pdf")

# ── Slow tasks ──
print("Computing age correlations for slow tasks...")
cor_dict_slow, stats_slow = prepare_correlation_posteriors(slow_tasks=True, n_samples=N_SAMPLES)
fig = plot_comparative_posteriors(
    cor_dict_slow,
    save_path=f"{FIGURES_DIR}figure3b_age_correlations_slow.pdf",
)
plt.close(fig)
print(f"  Saved {FIGURES_DIR}figure3b_age_correlations_slow.pdf")

# ── Posterior exclusion summary ──
print("\nPosterior sample exclusion (negative a or ndt):")
print(f"{'':10s} {'Task set':10s} {'Model':6s} {'Total':>12s} {'Excluded':>10s} {'%':>8s}")
print("-" * 55)
for task_label, stats in [("Fast", stats_fast), ("Slow", stats_slow)]:
    for model in ("ddm", "oum"):
        tot = stats[model]["total"]
        excl = stats[model]["excluded"]
        pct = 100.0 * excl / tot if tot > 0 else 0.0
        print(f"{'':10s} {task_label:10s} {model.upper():6s} {tot:>12,} {excl:>10,} {pct:>7.4f}%")

# Combined across fast + slow
print("-" * 55)
for model in ("ddm", "oum"):
    tot = stats_fast[model]["total"] + stats_slow[model]["total"]
    excl = stats_fast[model]["excluded"] + stats_slow[model]["excluded"]
    pct = 100.0 * excl / tot if tot > 0 else 0.0
    print(f"{'':10s} {'Combined':10s} {model.upper():6s} {tot:>12,} {excl:>10,} {pct:>7.4f}%")

# ── Table 3: Age correlations (LaTeX) ──
print("\nGenerating Table 3 (age correlations)...")

# Task label mapping: extract short name from path like "sfi_data/FF1"
def task_label(path):
    return os.path.basename(path).replace("sfi_data/", "")


def build_table_rows(cor_dict):
    """Build rows of median [95% CI] for each task and parameter."""
    rows = []
    params_all = ["v", "a", "ndt", "k"]

    tasks = sorted(cor_dict["ddm"].keys())
    for task in tasks:
        label = task_label(task)
        ddm = cor_dict["ddm"][task]
        oum = cor_dict["oum"][task]

        row = {"task": label}
        for p in ["v", "a", "ndt"]:
            samples = ddm[p]
            med = np.median(samples)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            row[f"ddm_{p}"] = f"${med:+.2f}$\\,[${lo:+.2f}$,\\,${hi:+.2f}$]"
        for p in params_all:
            samples = oum[p]
            med = np.median(samples)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            row[f"oum_{p}"] = f"${med:+.2f}$\\,[${lo:+.2f}$,\\,${hi:+.2f}$]"
        rows.append(row)
    return rows


fast_rows = build_table_rows(cor_dict_fast)
slow_rows = build_table_rows(cor_dict_slow)

# Write LaTeX table — two panels (DDM, OUM) stacked vertically
tex_path = "tableA_age_correlations_sfi.tex"
with open(tex_path, "w") as f:
    f.write(r"""\begin{table}[ht]
\centering
\caption{Median posterior age correlations [95\% CI] for DDM and OUM parameters across 18 RT tasks (Study 1).}
\label{tab:age_correlations_sfi}
\tiny
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.9}

\textbf{Panel A: DDM}\par\vspace{2pt}
\begin{tabular}{l ccc}
\toprule
Task & $\nu$ & $a$ & $\tau$ \\
\midrule
\multicolumn{4}{l}{\textit{Fast tasks}} \\
""")
    for row in fast_rows:
        f.write(f"{row['task']} & {row['ddm_v']} & {row['ddm_a']} & {row['ddm_ndt']}"
                r" \\" + "\n")
    f.write(r"\multicolumn{4}{l}{\textit{Slow tasks}} \\" + "\n")
    for row in slow_rows:
        f.write(f"{row['task']} & {row['ddm_v']} & {row['ddm_a']} & {row['ddm_ndt']}"
                r" \\" + "\n")
    f.write(r"""\bottomrule
\end{tabular}

\medskip
\textbf{Panel B: OUM}\par\vspace{2pt}
\begin{tabular}{l cccc}
\toprule
Task & $\nu$ & $a$ & $\tau$ & $k$ \\
\midrule
\multicolumn{5}{l}{\textit{Fast tasks}} \\
""")
    for row in fast_rows:
        f.write(f"{row['task']} & {row['oum_v']} & {row['oum_a']} & {row['oum_ndt']} & {row['oum_k']}"
                r" \\" + "\n")
    f.write(r"\multicolumn{5}{l}{\textit{Slow tasks}} \\" + "\n")
    for row in slow_rows:
        f.write(f"{row['task']} & {row['oum_v']} & {row['oum_a']} & {row['oum_ndt']} & {row['oum_k']}"
                r" \\" + "\n")

    f.write(r"""\bottomrule
\end{tabular}

\vspace{4pt}
\noindent\textit{Note.} Each cell shows the median posterior Pearson correlation with the 95\% credible interval in brackets. $\nu$ = drift rate, $a$ = boundary separation, $\tau$ = non-decision time, $k$ = self-excitation (OUM only).
\end{table}
""")

print(f"  Saved {tex_path}")

# Print summary to console
print("\nAge correlation summary (median posterior r):")
print("-" * 70)
print(f"{'Task':6s}  {'DDM v':>7s} {'DDM a':>7s} {'DDM t':>7s}  "
      f"{'OUM v':>7s} {'OUM a':>7s} {'OUM t':>7s} {'OUM k':>7s}")
print("-" * 70)
for label, rows in [("Fast", fast_rows), ("Slow", slow_rows)]:
    print(f"  {label} tasks:")
    for row in rows:
        vals = []
        for key in ["ddm_v", "ddm_a", "ddm_ndt", "oum_v", "oum_a", "oum_ndt", "oum_k"]:
            # Extract median from the formatted string
            med_str = row[key].split("$")[1]
            vals.append(f"{float(med_str):+.2f}")
        print(f"  {row['task']:6s}  {vals[0]:>7s} {vals[1]:>7s} {vals[2]:>7s}  "
              f"{vals[3]:>7s} {vals[4]:>7s} {vals[5]:>7s} {vals[6]:>7s}")

print("\nDone.")
