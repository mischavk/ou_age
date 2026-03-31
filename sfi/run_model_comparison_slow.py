"""Run model comparison (DDM vs OUM) for slow tasks.

Trains a BayesFlow ModelComparisonApproximator, validates on simulated data,
then applies to empirical slow task data. Saves Figure 2 and Appendix A1/A2.

Usage:
    python run_model_comparison_slow.py
"""

import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import gc
import keras
import seaborn as sns
import bayesflow as bf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfi_functions import (
    sfi_ddm_slow_prior, sfi_oum_slow_prior,
    sfi_likelihood_ddm, sfi_likelihood_oum,
    calculate_exceedance_probabilities,
)

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Training hyperparameters ──
NUM_BATCHES = 1000
BATCH_SIZE = 32
EPOCHS = 100

# ── 1. Create simulators ──
print("Creating simulators...")
sfi_ddm_simulator = bf.make_simulator([sfi_ddm_slow_prior, sfi_likelihood_ddm])
sfi_oum_simulator = bf.make_simulator([sfi_oum_slow_prior, sfi_likelihood_oum])

# ── 2. Set up adapter and network ──
print("Setting up adapter and network...")
adapter = (
    bf.Adapter()
    .as_set(["rts"])
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .drop(["v", "a", "ndt", "k"])
    .rename("model_indices", "inference_variables")
    .concatenate(["rts"], into="summary_variables")
)

simulator = bf.simulators.ModelComparisonSimulator(
    simulators=[sfi_ddm_simulator, sfi_oum_simulator],
    use_mixed_batches=True,
)

summary_network = bf.networks.SetTransformer(summary_dim=20)
classifier_network = bf.networks.MLP(widths=(256,) * 16, activation="relu")

approximator = bf.approximators.ModelComparisonApproximator(
    num_models=2,
    classifier_network=classifier_network,
    summary_network=summary_network,
    adapter=adapter,
    standardize="summary_variables",
)

learning_rate = keras.optimizers.schedules.CosineDecay(
    1e-4, decay_steps=EPOCHS * NUM_BATCHES, alpha=1e-5
)
optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
approximator.compile(optimizer=optimizer)

# ── 3. Train ──
print(f"Training model comparison network ({EPOCHS} epochs, {NUM_BATCHES} batches)...")
history = approximator.fit(
    epochs=EPOCHS,
    num_batches=NUM_BATCHES,
    batch_size=BATCH_SIZE,
    simulator=simulator,
    adapter=adapter,
)

# ── 4. Loss curve ──
print("Plotting loss curve...")
f = bf.diagnostics.plots.loss(history=history)
f.savefig(f"{FIGURES_DIR}loss_model_comparison_slow.pdf", bbox_inches="tight")
plt.close(f)

# ── 5. Validation on simulated data ──
print("Generating validation predictions (10,000 samples)...")
df_sampled = simulator.sample(10000)
pred_models = np.concatenate([
    approximator.predict(conditions={"rts": x})
    for x in np.array_split(df_sampled["rts"], 50)
], axis=0)

# Calibration (Appendix A1)
print("Plotting calibration...")
f = bf.diagnostics.plots.mc_calibration(
    pred_models=pred_models,
    true_models=df_sampled["model_indices"],
    model_names=["DDM", "OUM"],
)
f.savefig(f"{FIGURES_DIR}figureA1_calibration_slow.pdf", bbox_inches="tight")
plt.close(f)
print(f"  Saved {FIGURES_DIR}figureA1_calibration_slow.pdf")

# Confusion matrix (Appendix A2)
print("Plotting confusion matrix...")
f = bf.diagnostics.plots.mc_confusion_matrix(
    pred_models=pred_models,
    true_models=df_sampled["model_indices"],
    model_names=["DDM", "OUM"],
    normalize="true",
)
f.savefig(f"{FIGURES_DIR}figureA2_confusion_matrix_slow.pdf", bbox_inches="tight")
plt.close(f)
print(f"  Saved {FIGURES_DIR}figureA2_confusion_matrix_slow.pdf")

# ── 6. Empirical data analysis (Figure 2) ──
files = sorted([
    os.path.join("sfi_data", f)
    for f in os.listdir("sfi_data")
    if f.endswith(".txt") and f.startswith("S")
])

print(f"\nApplying to {len(files)} empirical slow tasks...")
print("Model probabilities for DDM vs. OUM on slow tasks:")

fig, axes = plt.subplots(3, 3, figsize=(6, 7), sharey=True)
axes = axes.T.flatten()

for ax, file in zip(axes, files):
    df = pd.read_csv(file, sep=" ")

    # Clean RTs
    df.loc[df["RT"] < 200, "RT"] = 0
    df.loc[df["RT"] > 10000, "RT"] = 0
    df["rt_mod"] = (df["RT"] * df["acc"].replace(0, -1)) / 1000

    # Keep only test trials, last 100 per person
    df = df[df["block"] == "test"]
    df_last100 = df.groupby("pp", group_keys=False).tail(100)

    grouped = df_last100.groupby("pp")
    n_persons = grouped.ngroups

    result = np.full((n_persons, 100, 1), np.nan, dtype=np.float32)
    for i, (_, group) in enumerate(grouped):
        rts = group["rt_mod"].values
        result[i, : len(rts), 0] = rts

    gc.collect()
    pred_models_empirical = approximator.predict(conditions={"rts": result})
    model_probs = np.median(pred_models_empirical, axis=0)

    task_name = os.path.basename(file).replace(".txt", "")
    print(f"  {task_name}: DDM={model_probs[0]:.3f}, OUM={model_probs[1]:.3f}")

    sns.violinplot(data=pred_models_empirical, ax=ax, cut=0)
    ax.set_title(task_name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DDM", "OUM"], rotation=0, fontsize=10)

plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}figure2_model_comparison_slow.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nFigure 2 (slow) saved to {FIGURES_DIR}figure2_model_comparison_slow.pdf")
print("Done.")
