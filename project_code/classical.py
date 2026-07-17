"""
Classical estimators for the latent network reliability model.
"""
import numpy as np


class NaiveEstimator:
    """Sample from prior, average failure indicator."""
    def __init__(self, model):
        self.model = model

    def estimate(self, n_samples, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        z = self.model.sample_prior(n_samples, rng=rng)
        weights = np.ones(n_samples)
        failures = np.array([1 - self.model.phi(zi) for zi in z])
        return np.mean(failures), np.var(failures, ddof=1), weights, failures


class TiltedEstimator:
    """
    Independent per-subsystem exponential tilting.
    Proposes Z_s ~ Bernoulli(1 - p_tilt[s]) i.e. failure prob p_tilt[s].
    Reweights by likelihood ratio.
    """
    def __init__(self, model, p_tilt):
        self.model = model
        self.p_tilt = np.asarray(p_tilt)

    def estimate(self, n_samples, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        z = self.model.sample_tilted(n_samples, self.p_tilt, rng=rng)
        lr = self.model.likelihood_ratio(z, self.p_tilt)
        failures = np.array([1 - self.model.phi(zi) for zi in z])
        weights = failures * lr
        return np.mean(weights), np.var(weights, ddof=1), weights, failures

    @staticmethod
    def grid_search(model, n_samples, p_grid, rng=None):
        """
        Grid search over a scalar tilt (same p_tilt for all subsystems).
        Returns best tilt and its VRF.
        """
        best_vrf = -1
        best_p = None
        best_est = None
        for p in p_grid:
            est = TiltedEstimator(model, np.full(model.m, p))
            mean, var, _, _ = est.estimate(n_samples, rng=rng)
            if var > 0:
                vrf = (model.exact_pfail()[0] * (1 - model.exact_pfail()[0])) / var
                if vrf > best_vrf:
                    best_vrf = vrf
                    best_p = p
                    best_est = est
        return best_p, best_vrf, best_est
