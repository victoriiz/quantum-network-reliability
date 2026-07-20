"""
pipeline/estimators.py -- LAYERS 2, 3, 4, and the CEILING.

NOTHING in this file mentions fault trees. It works against the model
contract (.p, .failmask, .p_fail, .component_bits, .n_components), so it
serves ANY static model unchanged. That model-agnosticism is the whole
point of the separation.
"""
import numpy as np
from scipy.optimize import minimize


# ===================== LAYER 2: the ideal proposal ====================
def ideal_proposal(model):
    """p*(x) = p(x) 1_fail(x) / P_fail -- the zero-variance target."""
    pstar = model.p * model.failmask / model.p_fail
    return pstar


# ===================== LAYER 3: proposal families =====================
def naive_proposal(model):
    return model.p.copy()                    # sampling the prior = naive MC


def tilt_proposal(model, p_tilt):
    """Per-component independent tilt with FAILURE probs p_tilt (len m).
    q(z) = prod_j [ (1-p_tilt_j) if z_j=1 else p_tilt_j ]."""
    bits = model.component_bits
    pw = np.where(bits == 1, 1 - p_tilt, p_tilt)
    return pw.prod(axis=1)


# ===================== LAYER 4: the audit =============================
def audit(model, q, n_samples=200_000, n_trials=20, seed=0):
    """Importance-sampling scorecard. Returns the full metric dict."""
    rng = np.random.default_rng(seed)
    idx = np.arange(model.n_states)
    num = model.p * model.failmask           # p(z)1_fail(z), precomputed
    ests = np.empty(n_trials)
    ess_frac = np.empty(n_trials)
    for t in range(n_trials):
        draw = rng.choice(idx, size=n_samples, p=q)
        w = np.where(q[draw] > 0, num[draw] / q[draw], 0.0)
        ests[t] = w.mean()
        s1, s2 = w.sum(), (w ** 2).sum()
        ess_frac[t] = (s1 * s1 / s2) / n_samples if s2 > 0 else 0.0
    mean = ests.mean()
    var_single = ests.var(ddof=1) * n_samples
    vrf = model.p_fail * (1 - model.p_fail) / var_single if var_single > 0 else np.inf
    bias = (mean - model.p_fail) / model.p_fail
    ci = 1.96 * ests.std(ddof=1) / np.sqrt(n_trials)
    mse = bias ** 2 * model.p_fail ** 2 + var_single / n_samples
    return {"mean": mean, "ci": ci, "bias": bias, "vrf": vrf,
            "ess": ess_frac.mean(), "mse": mse}


# ===================== CEILING: exact, no sampling ====================
def exact_variance(model, p_tilt):
    """Closed-form single-sample variance under tilt p_tilt.
    Var = sum_z [p(z)1_fail(z)]^2 / q(z) - P_fail^2, over ALL states."""
    q = tilt_proposal(model, p_tilt)
    num = (model.p * model.failmask) ** 2
    e_w2 = np.sum(np.where(q > 0, num / q, 0.0))
    return e_w2 - model.p_fail ** 2


def tilt_family_ceiling(model, seed=0):
    """Optimize the per-component tilt on EXACT variance -> the provable
    best VRF the whole product-tilt family can reach. No sampling.
    This is the HONEST OPPONENT that makes trained results falsifiable."""
    m = model.n_components
    x0 = np.clip(model.cfg.p_fail * 3, 1e-3, 0.5)      # sensible start
    # optimize log-variance for numerical stability; bounded in (0,1)
    obj = lambda v: np.log(exact_variance(model, np.clip(v, 1e-4, 0.999)))
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-9})
    p_star_tilt = np.clip(res.x, 1e-4, 0.999)
    var = exact_variance(model, p_star_tilt)
    vrf = model.p_fail * (1 - model.p_fail) / var
    return {"p_tilt": p_star_tilt, "vrf": vrf}