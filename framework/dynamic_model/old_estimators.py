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
    best VRF the whole product-tilt family can reach."""
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

def exact_traj_ceiling(model, proposal_kernel, param_grid):
    """Exact VRF ceiling over a family of trajectory proposals.
    For each params in param_grid, build Q = proposal_kernel(model, params)
    and evaluate the exact single-sample variance via the second-moment DP:
        M_0(x) = 0
        M_s(x) = sum_{y in F} R(x,y) + sum_{y not in F} R(x,y) M_{s-1}(y),
        with R(x,y) = P(x,y)^2 / Q(x,y),  stopping at first entry to F.
    Var = M_T(start) - P_fail^2.  Returns the best (params, vrf).
    """
    import numpy as np
    P, in_F, start, T, pT = (model.Pmat, model.in_F, model.start_idx,
                             model.T, model.p_fail)
    best = {"params": None, "vrf": -np.inf}
    for params in param_grid:
        Q = proposal_kernel(model, params)
        R = np.where(Q > 0, P * P / Q, 0.0)              # P^2/Q ratio matrix
        M = np.zeros(model.n_states)
        for _ in range(T):
            M = R[:, in_F == 1].sum(axis=1) + R[:, in_F == 0] @ M[in_F == 0]
            M[in_F == 1] = 0.0                           # never step from inside F
        var = M[start] - pT ** 2
        vrf = pT * (1 - pT) / var if var > 0 else np.inf
        if vrf > best["vrf"]:
            best = {"params": params, "vrf": float(vrf)}
    return best

