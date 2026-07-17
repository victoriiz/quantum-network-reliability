"""
Evaluation metrics for importance sampling estimators.
"""
import numpy as np


def compute_metrics(weights, exact_pfail, n_samples):
    """
    Compute bias, per-sample variance, VRF, ESS from importance weights.

    Parameters
    ----------
    weights : np.ndarray, shape (n_trials, n_samples)
        Importance weights per trial.
    exact_pfail : float
        Ground truth failure probability.
    n_samples : int
        Samples per trial.

    Returns
    -------
    dict with mean_estimate, bias, rel_bias, per_sample_var, VRF, ESS.
    """
    estimates = np.mean(weights, axis=1)
    mean_est = float(np.mean(estimates))
    bias = mean_est - exact_pfail
    rel_bias = bias / exact_pfail if exact_pfail > 0 else 0.0
    per_sample_var = float(np.var(estimates, ddof=1)) * len(estimates) / n_samples
    naive_var = exact_pfail * (1 - exact_pfail)
    vrf = naive_var / per_sample_var if per_sample_var > 0 else float('inf')

    # ESS over all trials pooled
    all_w = weights.flatten()
    ess = float((np.sum(all_w)**2) / np.sum(all_w**2)) if np.sum(all_w**2) > 0 else 0.0

    return {
        'mean_estimate': mean_est,
        'bias': bias,
        'rel_bias': rel_bias,
        'per_sample_var': per_sample_var,
        'VRF': vrf,
        'ESS': ess,
        'n_trials': len(estimates),
        'n_samples': n_samples
    }
