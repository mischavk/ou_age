"""
sfi_functions.py — Model definitions and analysis utilities for OUM/DDM comparison.

Implements the Ornstein-Uhlenbeck Model (OUM) and Drift Diffusion Model (DDM) using BayesFlow 2.0.10.

Models
------
DDM : x(t+dt) = x(t) + v·dt + ε               (ε ~ N(0, dt); k = 0)
OUM : x(t+dt) = x(t) + (v + k·x(t))·dt + ε   (self-excitation when k > 0)

Both models share the same simulator (``sfi_simulator_fun``); the OUM is obtained
by passing k > 0. Evidence accumulates between boundaries at ±a/2, starting at 0.
Positive simulated RTs indicate a correct (upper boundary) response; negative RTs
indicate an error (lower boundary) response. Trials exceeding 10 s or faster than
100 ms are treated as outliers and set to 0.

Task naming convention
----------------------
Fast tasks (F*) : FF1–FF3, FN1–FN3, FV1–FV3   (9 tasks)
Slow tasks (S*) : SF1–SF3, SN1–SN3, SV1–SV3   (9 tasks)
Suffix 1/2/3    : three independent tasks of the same category type.

Usage
-----
This module is imported by all SFI notebooks. 
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from scipy.special import digamma
from scipy.stats import wasserstein_distance

# Fixed RNG for reproducibility across all prior-sampling calls.
RNG = np.random.default_rng(2023)

# Quantile levels used for RT summary statistics.
QS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def _sigmoid(x: float) -> float:
    """Logistic sigmoid, clipped for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sample_sigmoid_a(mu: float, sigma: float, L: float) -> float:
    """Sample boundary separation from a sigmoid-transformed normal prior.

    Draws z ~ N(mu, sigma) and returns L * sigmoid(z), giving a bounded
    distribution on (0, L) whose center and spread are controlled by mu
    and sigma respectively.  Negative mu shifts mass toward zero (right-skewed).
    """
    z = RNG.normal(mu, sigma)
    return L * _sigmoid(z)


# ---------------------------------------------------------------------------
# Prior samplers
# ---------------------------------------------------------------------------

def sfi_ddm_fast_prior() -> dict:
    """Sample one draw from the DDM prior for fast tasks.

    Returns
    -------
    dict
        Keys: ``v`` (drift rate), ``a`` (boundary separation), ``ndt``
        (non-decision time).

    Notes
    -----
    Prior distributions::

        v   ~ Gamma(3.5, 1.0)
        a   ~ 8 * sigmoid(N(-1.0, 1.0))   [bounded (0, 8), mean ≈ 2.5]
        ndt ~ Uniform(0.1, 1.0)
    """
    v = RNG.gamma(3.5, 1.0)
    a = _sample_sigmoid_a(mu=-1.0, sigma=1.0, L=8.0)
    ndt = RNG.uniform(0.1, 1.0)
    return dict(v=v, a=a, ndt=ndt)


def sfi_oum_fast_prior() -> dict:
    """Sample one draw from the OUM prior for fast tasks.

    Returns
    -------
    dict
        Keys: ``v`` (drift rate), ``a`` (boundary separation), ``ndt``
        (non-decision time), ``k`` (self-excitation rate).

    Notes
    -----
    Prior distributions::

        v   ~ Gamma(3.5, 1.0)
        a   ~ 8 * sigmoid(N(0.0, 1.0))   [bounded (0, 8), mean ≈ 4.0, sd ≈ 1.7]
        ndt ~ Uniform(0.1, 1.0)
        k   ~ Gamma(4.0, 0.5) + 1.0   [mean=3.0, floor=1.0]
    """
    v = RNG.gamma(3.5, 1.0)
    a = _sample_sigmoid_a(mu=0.0, sigma=1.0, L=8.0)
    ndt = RNG.uniform(0.1, 1.0)
    k = RNG.gamma(4.0, 0.5) + 1.0
    return dict(v=v, a=a, ndt=ndt, k=k)


def sfi_ddm_slow_prior() -> dict:
    """Sample one draw from the DDM prior for slow tasks.

    Returns
    -------
    dict
        Keys: ``v`` (drift rate), ``a`` (boundary separation), ``ndt``
        (non-decision time).

    Notes
    -----
    Prior distributions::

        v   ~ Gamma(6.0, 0.25)   [tighter around lower mean for slow responses]
        a   ~ 10 * sigmoid(N(-1.3, 1.0))   [bounded (0, 10), mean ≈ 2.5, sd ≈ 1.7]
        ndt ~ Uniform(0.1, 3.0)  [longer NDT range for slow tasks]
    """
    v = RNG.gamma(6.0, 0.25)
    a = _sample_sigmoid_a(mu=-1.3, sigma=1.0, L=10.0)
    ndt = RNG.uniform(0.1, 3.0)
    return dict(v=v, a=a, ndt=ndt)


def sfi_oum_slow_prior() -> dict:
    """Sample one draw from the OUM prior for slow tasks.

    Returns
    -------
    dict
        Keys: ``v`` (drift rate), ``a`` (boundary separation), ``ndt``
        (non-decision time), ``k`` (self-excitation rate).

    Notes
    -----
    Prior distributions::

        v   ~ Gamma(6.0, 0.25)
        a   ~ 10 * sigmoid(N(1.3, 2.0))   [bounded (0, 10), mean ≈ 6.9, sd ≈ 2.4]
        ndt ~ Uniform(0.1, 3.0)
        k   ~ Gamma(4.0, 0.5) + 1.0   [mean=3.0, floor=1.0]
    """
    v = RNG.gamma(6.0, 0.25)
    a = _sample_sigmoid_a(mu=1.3, sigma=2.0, L=10.0)
    ndt = RNG.uniform(0.1, 3.0)
    k = RNG.gamma(4.0, 0.5) + 1.0
    return dict(v=v, a=a, ndt=ndt, k=k)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@njit
def sfi_simulator_fun(
    v: float,
    a: float,
    ndt: float,
    sndt: float = 0,
    sv: float = 0,
    k: float = 0.0,
) -> np.ndarray:
    """Simulate 100 RT trials via Euler-Maruyama integration (DDM or OUM).

    The same simulator implements both models:

    * **DDM** (k = 0): ``dx = v·dt + dW``
    * **OUM** (k > 0): ``dx = (v + k·x)·dt + dW``

    Evidence starts at 0 and is absorbed at ±a/2.

    Parameters
    ----------
    v : float
        Mean drift rate (signal strength toward the upper boundary).
    a : float
        Boundary separation (distance between upper and lower boundary).
    ndt : float
        Mean non-decision time (seconds).
    sndt : float, optional
        Half-range of uniform jitter added to ndt per trial. Default 0.
    sv : float, optional
        Standard deviation of trial-to-trial drift rate variability. Default 0.
    k : float, optional
        Self-excitation rate. Set to 0 for standard DDM. Default 0.

    Returns
    -------
    np.ndarray, shape (100,)
        Signed RTs: positive values indicate correct (upper boundary) responses,
        negative values indicate errors (lower boundary). Outlier trials
        (|RT| > 10 s or |RT| < 0.1 s) are set to 0.

    Notes
    -----
    Integration step: dt = 0.001 s. Maximum of 10,000 steps per trial (10 s).
    """
    out = np.zeros(100)

    for n in range(100):
        # Trial-specific drift rate
        v_trial = v + np.random.normal() * sv

        x = 0.0
        dt = 0.001
        n_steps = 0
        max_steps = 10000
        ndt_trial = ndt + np.random.uniform(-0.5 * sndt, 0.5 * sndt)

        # Euler-Maruyama integration until boundary hit or timeout
        while (x > -a / 2) and (x < a / 2) and (n_steps < max_steps):
            x += v_trial * dt + k * x * dt + np.sqrt(dt) * np.random.normal()
            n_steps += 1

        rt = n_steps * dt
        # Convention: positive RT = correct (upper), negative RT = error (lower)
        out[n] = rt + ndt_trial if x >= a / 2 else -rt - ndt_trial

        # Remove outliers
        if (abs(out[n]) > 10) or (abs(out[n]) < 0.2):
            out[n] = 0

    return out


# ---------------------------------------------------------------------------
# Likelihood wrappers
# ---------------------------------------------------------------------------

def sfi_likelihood_ddm(v: float, a: float, ndt: float, sndt: float = 0, sv: float = 0, k: float = 0.0) -> dict:
    """Return simulated RTs under the DDM (k fixed at 0).

    Wraps ``sfi_simulator_fun`` with k=0 for use as a BayesFlow likelihood.

    Parameters
    ----------
    v, a, ndt, sndt, sv : float
        Model parameters (see ``sfi_simulator_fun``).
    k : float
        Ignored for the DDM; included for API compatibility.

    Returns
    -------
    dict
        ``{"rts": np.ndarray}`` — 100 signed RTs.
    """
    rts = sfi_simulator_fun(v=v, a=a, ndt=ndt, sndt=sndt, sv=sv, k=k)
    return dict(rts=rts)


def sfi_likelihood_oum(v: float, a: float, ndt: float, k: float, sndt: float = 0, sv: float = 0) -> dict:
    """Return simulated RTs under the OUM (k > 0 enables self-excitation).

    Wraps ``sfi_simulator_fun`` for use as a BayesFlow likelihood.

    Parameters
    ----------
    v, a, ndt, sndt, sv, k : float
        Model parameters (see ``sfi_simulator_fun``).

    Returns
    -------
    dict
        ``{"rts": np.ndarray}`` — 100 signed RTs.
    """
    rts = sfi_simulator_fun(v=v, a=a, ndt=ndt, sndt=sndt, sv=sv, k=k)
    return dict(rts=rts)


# ---------------------------------------------------------------------------
# Data summary utilities
# ---------------------------------------------------------------------------

def summarize_empirical_data(rt: np.ndarray) -> dict:
    """Compute quantile summaries and full RT arrays for correct and error trials.

    Parameters
    ----------
    rt : np.ndarray
        1-D array of signed RTs. Positive values are correct responses,
        negative values are errors.

    Returns
    -------
    dict with keys:
        ``c_qs`` : np.ndarray, shape (5,)
            Quantiles at [0.1, 0.3, 0.5, 0.7, 0.9] for correct RTs.
            All NaN if no correct trials.
        ``c_bin`` : np.ndarray
            Full array of correct RTs (for Wasserstein distance).
        ``e_qs`` : np.ndarray, shape (5,)
            Quantiles for error RTs (absolute values).
        ``e_bin`` : np.ndarray
            Full array of error RTs.
    """
    out = {}

    # Correct trials
    mask = rt > 0
    x = rt[mask]
    if x.size > 0:
        out["c_qs"] = np.quantile(x, QS)
        out["c_bin"] = x.copy()
    else:
        out["c_qs"] = np.full(QS.shape[0], np.nan)
        out["c_bin"] = np.array([])

    # Error trials
    mask = rt < 0
    x = rt[mask]
    if x.size > 0:
        out["e_qs"] = np.quantile(x, QS)
        out["e_bin"] = x.copy()
    else:
        out["e_qs"] = np.full(QS.shape[0], np.nan)
        out["e_bin"] = np.array([])

    return out


@njit(parallel=True)
def compute_rmses(
    sim_rts: np.ndarray,
    emp_c_qs: np.ndarray,
    emp_e_qs: np.ndarray,
) -> np.ndarray:
    """Compute per-quantile RMSE between simulated and empirical RTs.

    Calculates RMSE at each of the 5 quantile levels (0.1, 0.3, 0.5, 0.7, 0.9)
    separately for correct and error trials, then averages across posterior samples.

    Parameters
    ----------
    sim_rts : np.ndarray, shape (n_samples, 100)
        Simulated RTs from posterior predictive samples.
    emp_c_qs : np.ndarray, shape (5,)
        Empirical quantiles for correct RTs.
    emp_e_qs : np.ndarray, shape (5,)
        Empirical quantiles for error RTs.

    Returns
    -------
    np.ndarray, shape (10,)
        Mean RMSE across samples: indices 0–4 = correct quantiles (Q1–Q9),
        indices 5–9 = error quantiles (Q1–Q9).
    """
    n_samples = sim_rts.shape[0]
    n_metrics = 10
    rmses = np.empty((n_samples, n_metrics))
    qs_len = len(emp_c_qs)

    for s in prange(n_samples):
        rt = sim_rts[s]

        # Correct trials
        sim_rt = rt[rt > 0]
        if sim_rt.size > 0:
            sim_qs = np.quantile(sim_rt, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        else:
            sim_qs = np.full(qs_len, np.nan)
        for i in range(qs_len):
            rmses[s, i] = np.sqrt((emp_c_qs[i] - sim_qs[i]) ** 2)

        # Error trials
        sim_rt = rt[rt < 0]
        if sim_rt.size > 0:
            sim_qs = np.quantile(sim_rt, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        else:
            sim_qs = np.full(qs_len, np.nan)
        for i in range(qs_len):
            rmses[s, i + 5] = np.sqrt((emp_e_qs[i] - sim_qs[i]) ** 2)

    # Average across posterior samples (NaN-safe)
    mean_rmses = np.empty(n_metrics)
    for j in range(n_metrics):
        mean_rmses[j] = nanmean_numba(rmses[:, j])
    return mean_rmses


@njit
def nanmean_numba(a: np.ndarray) -> float:
    """Compute the mean of ``a``, ignoring NaN values (Numba-compatible).

    Parameters
    ----------
    a : np.ndarray, shape (n,)
        Input array.

    Returns
    -------
    float
        Mean of non-NaN elements, or NaN if all elements are NaN.
    """
    s = 0.0
    n = 0
    for i in range(a.shape[0]):
        val = a[i]
        if not np.isnan(val):
            s += val
            n += 1
    return s / n if n > 0 else np.nan


def safe_wasserstein(emp_bin: np.ndarray, sim_bin: np.ndarray) -> float:
    """Compute the Wasserstein distance, returning NaN for empty arrays.

    Parameters
    ----------
    emp_bin : np.ndarray
        Empirical RT distribution.
    sim_bin : np.ndarray
        Simulated RT distribution.

    Returns
    -------
    float
        Wasserstein-1 distance, or NaN if either array is empty.
    """
    if emp_bin.size == 0 or sim_bin.size == 0:
        return np.nan
    return wasserstein_distance(emp_bin, sim_bin)


# ---------------------------------------------------------------------------
# Model comparison utilities
# ---------------------------------------------------------------------------

def calculate_exceedance_probabilities(
    pmp_matrix: np.ndarray,
    iterations: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate exceedance probabilities via variational Bayes on a Dirichlet model.

    Fits a Dirichlet posterior to the posterior model probability (PMP) matrix
    using a mean-field variational Bayes algorithm, then Monte-Carlo estimates
    the probability that each model has the highest frequency in the population.

    Parameters
    ----------
    pmp_matrix : np.ndarray, shape (n_subjects, n_models)
        Per-subject posterior model probabilities.
    iterations : int, optional
        Number of Monte-Carlo samples from the posterior Dirichlet for EP
        estimation. Default 10 000.

    Returns
    -------
    eps : np.ndarray, shape (n_models,)
        Exceedance probability for each model.
    alpha_post : np.ndarray, shape (n_models,)
        Posterior Dirichlet concentration parameters.
    """
    n_subjects, n_models = pmp_matrix.shape
    alpha_0 = np.ones(n_models)  # Uniform prior
    alpha = np.copy(alpha_0)

    # Variational Bayes: iterate until convergence (fixed 100 steps)
    for _ in range(100):
        exp_log_r = digamma(alpha) - digamma(np.sum(alpha))
        weights = np.exp(exp_log_r)

        g = pmp_matrix * weights
        g /= g.sum(axis=1)[:, np.newaxis]

        alpha = alpha_0 + g.sum(axis=0)

    # Monte-Carlo estimate of exceedance probabilities
    samples = np.random.dirichlet(alpha, size=iterations)
    eps = (samples.argmax(axis=1)[:, None] == np.arange(n_models)).mean(axis=0)

    return eps, alpha


# ---------------------------------------------------------------------------
# Analysis: age–parameter correlations
# ---------------------------------------------------------------------------

def prepare_correlation_posteriors(
    slow_tasks: bool = True,
    n_samples: int = 1000,
) -> tuple:
    """Compute posterior distributions of Pearson correlations between each
    model parameter and age, for all 9 fast or slow tasks.

    For each task and each posterior sample, computes the correlation between
    participant-level parameter estimates and their ages. Posterior samples with
    theoretically impossible values (negative boundary separation or non-decision
    time) are excluded before computing correlations.

    Parameters
    ----------
    slow_tasks : bool, optional
        If True, processes slow tasks (S* files) and loads ``*_sfi_slow_*``
        posteriors. If False, processes fast tasks (F* files). Default True.
    n_samples : int, optional
        Number of posterior samples to draw per parameter per task.
        Larger values give smoother correlation distributions. Default 1000.

    Returns
    -------
    cor_dict : dict
        Nested dictionary with structure::

            {
              "ddm": {task_path: {param_name: np.ndarray(n_samples)}},
              "oum": {task_path: {param_name: np.ndarray(n_samples)}},
            }

        where ``task_path`` is the full path to the task ``.txt`` file and
        ``param_name`` is one of ``"v"``, ``"a"``, ``"ndt"`` (DDM) or
        additionally ``"k"`` (OUM).
    excl_stats : dict
        Exclusion statistics with keys ``"ddm"`` and ``"oum"``, each containing
        ``"total"`` (total participant×sample pairs) and ``"excluded"`` (number
        excluded due to negative ``a`` or ``ndt``).

    Notes
    -----
    Posterior samples are loaded from ``sfi_data/*_sfi_{speed}_{model}_estimates.npy``
    files. Each file contains a dict mapping parameter names to arrays of shape
    ``(n_participants, n_samples, 1)``.
    """
    cor_dict = {"ddm": {}, "oum": {}}
    excl_stats = {
        "ddm": {"total": 0, "excluded": 0},
        "oum": {"total": 0, "excluded": 0},
    }

    questionnaires = pd.read_csv("sfi_data/estimates/questionnaires.csv", sep=" ")
    age = questionnaires["age"]
    ids = questionnaires["pp"]

    prefix = "S" if slow_tasks else "F"
    speed = "slow" if slow_tasks else "fast"

    files = sorted([
        os.path.join("sfi_data", f)
        for f in os.listdir("sfi_data")
        if f.endswith(".txt") and f.startswith(prefix)
    ])

    correlation_results_ddm = {file: {} for file in files}
    correlation_results_oum = {file: {} for file in files}

    for file in files:
        ids_in_task = pd.read_csv(file, sep=" ")["pp"].unique()
        age_in_task = age[ids.isin(ids_in_task)].values

        task = file.replace(".txt", "")

        posterior_samples_ddm = np.load(
            file.replace(".txt", f"_sfi_{speed}_ddm_estimates.npy"),
            allow_pickle=True,
        )[()]

        posterior_samples_oum = np.load(
            file.replace(".txt", f"_sfi_{speed}_oum_estimates.npy"),
            allow_pickle=True,
        )[()]

        # Build validity masks: sample is valid if a >= 0 AND ndt >= 0
        # Shape: (n_participants, n_samples)
        valid_ddm = (
            (posterior_samples_ddm["a"][:, :, 0] >= 0) &
            (posterior_samples_ddm["ndt"][:, :, 0] >= 0)
        )
        valid_oum = (
            (posterior_samples_oum["a"][:, :, 0] >= 0) &
            (posterior_samples_oum["ndt"][:, :, 0] >= 0)
        )

        excl_stats["ddm"]["total"] += valid_ddm.size
        excl_stats["ddm"]["excluded"] += int((~valid_ddm).sum())
        excl_stats["oum"]["total"] += valid_oum.size
        excl_stats["oum"]["excluded"] += int((~valid_oum).sum())

        # DDM: for each parameter, compute correlation using only valid samples
        for parameter, param_samples in posterior_samples_ddm.items():
            n = param_samples.shape[1]
            correlation_results_ddm[file][parameter] = np.zeros(n)
            for s in range(n):
                mask = valid_ddm[:, s]
                if mask.sum() < 3:
                    correlation_results_ddm[file][parameter][s] = np.nan
                    continue
                values = np.squeeze(param_samples[:, s, 0])[mask]
                correlation_results_ddm[file][parameter][s] = np.corrcoef(
                    values, age_in_task[mask]
                )[1, 0]

        # OUM: same as DDM but includes k
        for parameter, param_samples in posterior_samples_oum.items():
            n = param_samples.shape[1]
            correlation_results_oum[file][parameter] = np.zeros(n)
            for s in range(n):
                mask = valid_oum[:, s]
                if mask.sum() < 3:
                    correlation_results_oum[file][parameter][s] = np.nan
                    continue
                values = np.squeeze(param_samples[:, s, 0])[mask]
                correlation_results_oum[file][parameter][s] = np.corrcoef(
                    values, age_in_task[mask]
                )[1, 0]

        cor_dict["ddm"][task] = correlation_results_ddm[file]
        cor_dict["oum"][task] = correlation_results_oum[file]

    return cor_dict, excl_stats


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_comparative_posteriors(
    cor_dict: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot posterior correlation distributions (DDM vs. OUM) across 9 tasks.

    Creates a 3×3 grid of split violin plots. Each subplot shows the posterior
    distribution of Pearson correlations with age for each model parameter,
    comparing DDM (blue) and OUM (orange) side-by-side.

    Parameters
    ----------
    cor_dict : dict
        Output of ``prepare_correlation_posteriors``. Must contain keys
        ``"ddm"`` and ``"oum"``, each mapping task paths to parameter dicts.
        Expects exactly 9 tasks.
    save_path : str or None, optional
        If provided, saves the figure to this path before displaying.
        Supports any format recognised by matplotlib (e.g. ``".pdf"``,
        ``".png"``). Default None (display only).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Notes
    -----
    Assumes exactly 9 tasks. OUM includes all DDM parameters plus ``k``.
    """
    param_order = ["v", "a", "ndt", "k"]
    param_labels = {"v": "ν", "a": "a", "ndt": "τ", "k": "k"}
    palette = {"DDM": "tab:blue", "OUM": "tab:orange"}

    tasks = sorted(cor_dict["ddm"].keys())

    fig, axs = plt.subplots(3, 3, figsize=(18, 15), squeeze=False)
    axs = axs.flatten()

    legend_handles, legend_labels = None, None

    for i, task in enumerate(tasks):
        ax = axs[i]
        records = []

        for param in param_order:
            if param in cor_dict["ddm"][task]:
                samples = np.asarray(cor_dict["ddm"][task][param]).ravel()
                records.extend((param, v, "DDM") for v in samples)
            if param in cor_dict["oum"][task]:
                samples = np.asarray(cor_dict["oum"][task][param]).ravel()
                records.extend((param, v, "OUM") for v in samples)

        df = pd.DataFrame(records, columns=["parameter", "value", "model"])

        sns.violinplot(
            data=df,
            x="parameter",
            y="value",
            hue="model",
            order=param_order,
            split=True,
            inner="quartile",
            palette=palette,
            ax=ax,
        )

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

        ax.set_xticks(range(len(param_order)))
        ax.set_xticklabels([param_labels[p] for p in param_order], fontsize=16)
        ax.set_xlabel("Parameter", fontsize=14)
        ax.set_ylabel("Correlation with age", fontsize=14)
        ax.set_ylim(-0.5, 0.7)
        ax.axhline(0, color="black", linewidth=1.5)
        ax.set_title(f"Task: {os.path.basename(task)}", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        fontsize=14,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig


def plot_model_comparison_empirical(
    results: dict,
    model_names: list | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot per-task posterior model probabilities for empirical data.

    Creates a 3×3 grid of violin plots, one per task, showing the distribution
    of posterior model probabilities across participants for DDM vs. OUM.

    Parameters
    ----------
    results : dict
        Mapping from task name (str) to an array of shape
        ``(n_participants, n_models)`` containing posterior model probabilities.
        Must contain exactly 9 tasks.
    model_names : list of str, optional
        Display names for the models. Default ``["DDM", "OUM"]``.
    save_path : str or None, optional
        If provided, saves the figure to this path before displaying.
        Default None (display only).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Examples
    --------
    >>> # After running the model comparison approximator on all tasks:
    >>> results = {task: pred_models_empirical for task, pred_models_empirical in ...}
    >>> plot_model_comparison_empirical(results, save_path="figures/figure2.pdf")
    """
    if model_names is None:
        model_names = ["DDM", "OUM"]

    tasks = sorted(results.keys())
    fig, axes = plt.subplots(3, 3, figsize=(9, 10), sharey=True)
    axes = axes.T.flatten()

    for ax, task in zip(axes, tasks):
        pmp_array = results[task]
        sns.violinplot(data=pmp_array, ax=ax, cut=0)
        ax.set_title(os.path.basename(task).replace(".txt", ""), fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(model_names, rotation=0, fontsize=10)
        ax.set_ylabel("Posterior model probability", fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return fig
