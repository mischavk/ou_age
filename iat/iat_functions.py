"""iat_functions.py — Model definitions and utilities for IAT DDM/OUM comparison.

Implements the Ornstein-Uhlenbeck Model (OUM) and Drift Diffusion Model (DDM) for
reaction-time analysis of Implicit Association Test (IAT) data using BayesFlow 2.0.12.

Models
------
DDM : x(t+dt) = x(t) + v·dt + ε        (ε ~ N(0, dt))
OUM : x(t+dt) = x(t) + (v + k·x(t))·dt + ε   (self-excitation when k > 0)

IAT-specific features
---------------------
- Two experimental conditions (congruent / incongruent), each with own v and a
- 120 trials per person (60 per condition, 30 per stimulus type)
- Separate non-decision times for correct and error responses
- Evidence starts at 0, boundaries at ±a/2 (matching SFI parameterization)
"""

import numpy as np
from numba import njit, prange

RNG = np.random.default_rng(2023)

QS = np.array([0.1, 0.3, 0.7, 0.9])


# ---------------------------------------------------------------------------
# Sigmoid helpers 
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Logistic sigmoid, clipped for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sample_sigmoid_a(mu: float, sigma: float, L: float) -> float:
    """Sample boundary separation from a sigmoid-transformed normal prior.

    Draws z ~ N(mu, sigma) and returns L * sigmoid(z), giving a bounded
    distribution on (0, L) whose center and spread are controlled by mu
    and sigma respectively.
    """
    z = RNG.normal(mu, sigma)
    return L * _sigmoid(z)


# ---------------------------------------------------------------------------
# Prior functions
# ---------------------------------------------------------------------------

def iat_oum_prior_fun() -> dict:
    """Sample one draw from the OUM prior for IAT.

    Returns
    -------
    dict
        Keys: ``drifts`` (2,), ``thresholds`` (2,), ``ndt_correct``,
        ``ndt_error``, ``k``.

    Notes
    -----
    Prior distributions::

        drifts      ~ Gamma(3.5, 1.0) × 2
        thresholds  ~ 8·sigmoid(N(-0.25, 1.0)) × 2   [bounded (0, 8)]
        ndt_correct ~ Uniform(0.1, 1.0)
        ndt_error   ~ Gamma(2.0, 0.3)
        k           ~ Gamma(8.0, 0.5)               [mean = 4.0]
    """
    drifts = RNG.gamma(3.5, 1.0, size=2)
    thresholds = np.array([
        _sample_sigmoid_a(mu=-0.25, sigma=1.0, L=8.0),
        _sample_sigmoid_a(mu=-0.25, sigma=1.0, L=8.0),
    ])
    ndt_correct = RNG.uniform(0.1, 1.0)
    ndt_error = RNG.gamma(2.0, 0.3)
    k = RNG.gamma(8.0, 0.5)

    return dict(
        drifts=drifts,
        thresholds=thresholds,
        ndt_correct=ndt_correct,
        ndt_error=ndt_error,
        k=k,
    )


def iat_ddm_prior_fun() -> dict:
    """Sample one draw from the DDM prior for IAT.

    Returns
    -------
    dict
        Keys: ``drifts`` (2,), ``thresholds`` (2,), ``ndt_correct``,
        ``ndt_error``.

    Notes
    -----
    Prior distributions::

        drifts      ~ Gamma(3.5, 1.0) × 2
        thresholds  ~ 8·sigmoid(N(-1.6, 1.0)) × 2   [bounded (0, 8)]
        ndt_correct ~ Uniform(0.1, 1.0)
        ndt_error   ~ Gamma(2.0, 0.3)
    """
    drifts = RNG.gamma(3.5, 1.0, size=2)
    thresholds = np.array([
        _sample_sigmoid_a(mu=-1.6, sigma=1.0, L=8.0),
        _sample_sigmoid_a(mu=-1.6, sigma=1.0, L=8.0),
    ])
    ndt_correct = RNG.uniform(0.1, 1.0)
    ndt_error = RNG.gamma(2.0, 0.3)

    return dict(
        drifts=drifts,
        thresholds=thresholds,
        ndt_correct=ndt_correct,
        ndt_error=ndt_error,
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@njit
def iat_trial(v, a, ndt_correct, ndt_error, k=0.0):
    """Simulate a single IAT trial via Euler-Maruyama integration.

    Evidence starts at 0 and is absorbed at ±a/2 (matching SFI parameterization).
    Correct responses hit the upper boundary (+a/2), errors hit the lower (-a/2).
    """
    x = 0.0
    dt = 0.001
    n_steps = 0
    max_steps = 10000

    while (x > -a / 2) and (x < a / 2) and (n_steps < max_steps):
        x += v * dt + k * x * dt + np.sqrt(dt) * np.random.normal()
        n_steps += 1

    rt = n_steps * dt
    return rt + ndt_correct if x >= a / 2 else -rt - ndt_error


@njit
def iat_simulator_fun(drifts, thresholds, ndt_correct, ndt_error, k=0.0):
    """Simulate 120 IAT trials (60 congruent + 60 incongruent) for one person.

    Returns
    -------
    np.ndarray, shape (120, 4)
        Columns: [signed_RT, missing_flag, condition_type, stimulus_type]
    """
    num_obs = 120
    obs_per_condition = 60

    condition_type = np.arange(2)
    condition_type = np.repeat(condition_type, obs_per_condition)

    stimulus_type = np.concatenate((
        np.zeros(30), np.ones(30),   # condition 0: congruent
        np.zeros(30), np.ones(30),   # condition 1: incongruent
    ))

    v = drifts[0:2]
    a = thresholds[0:2]

    out = np.zeros(num_obs)

    for n in range(num_obs):
        out[n] = iat_trial(v[condition_type[n]], a[condition_type[n]],
                           ndt_correct, ndt_error, k)
        if abs(out[n]) > 10.0:
            out[n] = 0
        if abs(out[n]) < 0.2:
            out[n] = 0

    missings = np.expand_dims(np.zeros(out.shape[0]), 1)
    missings[out == 0] = 1

    out = np.expand_dims(out, 1)
    condition_type = np.expand_dims(condition_type, 1)
    stimulus_type = np.expand_dims(stimulus_type, 1)
    out = np.concatenate((out, missings, condition_type, stimulus_type), axis=1)

    return out


# ---------------------------------------------------------------------------
# Likelihood wrapper
# ---------------------------------------------------------------------------

def iat_likelihood(drifts, thresholds, ndt_correct, ndt_error, k=0.0):
    """Likelihood wrapper for BayesFlow: returns 'out' (the trial array)."""
    out = iat_simulator_fun(drifts, thresholds, ndt_correct, ndt_error, k)
    return dict(out=out)


# ---------------------------------------------------------------------------
# PPC utilities
# ---------------------------------------------------------------------------

def rmse_ignore_nan(y_true, y_pred):
    """Calculate RMSE, ignoring NaN values."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    return np.sqrt(np.mean((y_true_filtered - y_pred_filtered) ** 2))


@njit
def nanmean_numba(a):
    """Numba-safe nanmean for 1-D arrays."""
    s = 0.0
    n = 0
    for i in range(a.shape[0]):
        val = a[i]
        if not np.isnan(val):
            s += val
            n += 1
    return s / n if n > 0 else np.nan


@njit(parallel=True)
def compute_rmses_parallel_safe(sim_rts, sim_cond,
                                emp_c_cong_med, emp_c_cong_qs,
                                emp_e_cong_med, emp_e_cong_qs,
                                emp_c_inc_med, emp_c_inc_qs,
                                emp_e_inc_med, emp_e_inc_qs):
    """Compute RMSE metrics for congruent and incongruent trials (correct + error).

    Returns 20 metrics, averaged across PPC samples:
      congruent   [median_c, q1_c, q3_c, q7_c, q9_c, median_e, q1_e, q3_e, q7_e, q9_e]
      incongruent [median_c, q1_c, q3_c, q7_c, q9_c, median_e, q1_e, q3_e, q7_e, q9_e]
    (indices 0-9 congruent, 10-19 incongruent).
    """
    n_samples, n_trials = sim_rts.shape
    n_metrics = 20
    rmses = np.empty((n_samples, n_metrics))
    qs_len = len(emp_c_cong_qs)
    qgrid = np.array([0.1, 0.3, 0.7, 0.9])

    for s in prange(n_samples):
        rt = sim_rts[s]
        cond = sim_cond[s]

        c = rt > 0
        e = rt < 0
        # cond_val 0 = congruent (offset 0), 1 = incongruent (offset 10)
        for ci in range(2):
            base = ci * 10
            condm = cond == ci
            if ci == 0:
                ec_med = emp_c_cong_med; ec_qs = emp_c_cong_qs
                ee_med = emp_e_cong_med; ee_qs = emp_e_cong_qs
            else:
                ec_med = emp_c_inc_med; ec_qs = emp_c_inc_qs
                ee_med = emp_e_inc_med; ee_qs = emp_e_inc_qs

            # correct
            sim_rt = rt[c & condm]
            if sim_rt.size > 0:
                sim_med = np.median(sim_rt)
                sim_qs = np.quantile(sim_rt, qgrid)
            else:
                sim_med = np.nan
                sim_qs = np.full(qs_len, np.nan)
            rmses[s, base + 0] = np.sqrt((ec_med - sim_med) ** 2)
            for i in range(qs_len):
                rmses[s, base + i + 1] = np.sqrt((ec_qs[i] - sim_qs[i]) ** 2)

            # error
            sim_rt = rt[e & condm]
            if sim_rt.size > 0:
                sim_med = np.median(sim_rt)
                sim_qs = np.quantile(sim_rt, qgrid)
            else:
                sim_med = np.nan
                sim_qs = np.full(qs_len, np.nan)
            rmses[s, base + 5] = np.sqrt((ee_med - sim_med) ** 2)
            for i in range(qs_len):
                rmses[s, base + i + 6] = np.sqrt((ee_qs[i] - sim_qs[i]) ** 2)

    # posterior mean
    mean_rmses = np.empty(n_metrics)
    for j in range(n_metrics):
        mean_rmses[j] = nanmean_numba(rmses[:, j])
    return mean_rmses


def summarize_empirical_safe(emp):
    """Compute medians and quantiles for congruent and incongruent conditions."""
    rt = emp[:, 0]
    cond = emp[:, 2]
    out = {}

    for label, rt_sign, cond_val in [
        ("c_cong", 1, 0), ("e_cong", -1, 0),
        ("c_inc", 1, 1), ("e_inc", -1, 1),
    ]:
        if rt_sign > 0:
            mask = (rt > 0) & (cond == cond_val)
        else:
            mask = (rt < 0) & (cond == cond_val)
        x = rt[mask]
        if x.size > 0:
            out[f"{label}_med"] = np.median(x)
            out[f"{label}_qs"] = np.quantile(x, QS)
        else:
            out[f"{label}_med"] = np.nan
            out[f"{label}_qs"] = np.full(QS.shape[0], np.nan)

    return out


def summarize_empirical_bins(emp):
    """Compute medians, quantiles"""
    rt = emp[:, 0]
    cond = emp[:, 2]
    out = {}

    for label, rt_sign, cond_val in [
        ("c_cong", 1, 0), ("e_cong", -1, 0),
        ("c_inc", 1, 1), ("e_inc", -1, 1),
    ]:
        if rt_sign > 0:
            mask = (rt > 0) & (cond == cond_val)
        else:
            mask = (rt < 0) & (cond == cond_val)
        x = rt[mask]
        if x.size > 0:
            out[f"{label}_med"] = np.median(x)
            out[f"{label}_qs"] = np.quantile(x, QS)
        else:
            out[f"{label}_med"] = np.nan
            out[f"{label}_qs"] = np.full(QS.shape[0], np.nan)

    return out
