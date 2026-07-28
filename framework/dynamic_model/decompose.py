"""
Decomposing the optimality gap: how much is TIME, how much is STATE?
Goes in: framework/dynamic_model/decompose.py

MOTIVATION. The zero-variance proposal q*_t(x->y) ~ P(x,y) h_{t-1}(y) is optimal
because it depends on two things at once: the current state x and the time
remaining t. The committor / TD literature is INFINITE-HORIZON, where there is
no time dependence to separate -- so the question "how much of the achievable
variance reduction comes from time-dependence versus state-dependence?" cannot
even be posed there. It is a finite-horizon-only question, and the exact
second-moment recursion answers it without sampling.

THE LADDER. Each rung adds one kind of expressiveness:

    rung  family                params   time?  component?
    ------------------------------------------------------
     1    scalar                  1       no      no
     2    per_component           N       no      yes
     3    time_scalar             T       yes     no
     4    time_per_component     N*T      yes     yes
     5    h-transform             --      yes     FULL STATE (not just
                                                  component identity)

Rung 3 vs rung 2 isolates the value of knowing the clock.
Rung 4 vs rung 3 isolates the value of knowing which component.
Rung 5 vs rung 4 is the residual: everything lost by staying product-form,
i.e. by tilting each component independently instead of conditioning on the
full joint state.

Rungs 1-4 are nested, so their ceilings MUST be non-decreasing. Warm-starting
each from the solved smaller rung enforces it and makes a converged run
self-certifying.
"""

import numpy as np
from scipy.optimize import minimize

from proposals import build_kernel, rate_matrices


# --------------------------------------------------------------------------
# exact variance for a TIME-VARYING proposal
# --------------------------------------------------------------------------

def exact_traj_variance_tv(model, Q_seq, check_support=True):
    """Second-moment recursion with a different kernel at every step.

        M_0(x) = 0
        M_s(x) = sum_{y in F} R_t(x,y) + sum_{y not in F} R_t(x,y) M_{s-1}(y)
                 where t = T - s  and  R_t = P^2 / Q_t

    s counts steps REMAINING, so s = 1 is the final step (t = T-1) and s = T is
    the first step (t = 0). The recursion is built backward from the end of the
    mission, which is why the index runs T-s. With all Q_t equal this reduces
    exactly to the time-homogeneous recursion in ceiling.py.
    """
    P, in_F, T = model.Pmat, model.in_F, model.T
    if len(Q_seq) != T:
        raise ValueError(f"need T={T} kernels, got {len(Q_seq)}")

    fail_cols = in_F == 1
    safe_cols = ~fail_cols

    Rs, hits = [], []
    for Q in Q_seq:
        if check_support and np.any((P > 0) & (Q <= 0)):
            return np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(Q > 0, P * P / Q, 0.0)
        Rs.append(R)
        hits.append(R[:, fail_cols].sum(axis=1))

    M = np.zeros(model.n_states)
    for s in range(1, T + 1):
        t = T - s
        M = hits[t] + Rs[t][:, safe_cols] @ M[safe_cols]
        M[fail_cols] = 0.0

    return float(M[model.start_idx] - model.p_fail ** 2)


# --------------------------------------------------------------------------
# the five rungs
# --------------------------------------------------------------------------

def class_map(model):
    """Indices of the exchangeability classes of the components.

    Components with identical (c_i, a0_i, b0_i, gamma_i, eta_i) are
    interchangeable, so EVERY optimum assigns them the same tilt. Optimising
    one parameter per class instead of per component is therefore EXACT, not an
    approximation -- and it is what makes the full time-by-component rung
    computable at all: the Delta instance has 8 components but only 2 classes
    (the 8-GPU nodes and the 4-GPU nodes), so the top rung needs 2T parameters
    instead of 8T.

    Verified empirically: the free 8-parameter optimum respects these classes
    to 8.6e-6.
    """
    cfg = model.cfg
    key = np.stack([cfg.c, cfg.a0, cfg.b0, cfg.gamma, cfg.eta], axis=1)
    _, inv = np.unique(np.round(key, 12), axis=0, return_inverse=True)
    return inv, int(inv.max()) + 1


def _kernels_scalar(model, theta, rates):
    N = model.cfg.N
    Q = build_kernel(model, np.full(N, np.exp(theta[0])), None, rates=rates)
    return [Q] * model.T


def _kernels_class(model, theta, rates):
    inv, _ = class_map(model)
    Q = build_kernel(model, np.exp(np.asarray(theta))[inv], None, rates=rates)
    return [Q] * model.T


def _kernels_time_scalar(model, theta, rates):
    N = model.cfg.N
    return [build_kernel(model, np.full(N, np.exp(theta[t])), None, rates=rates)
            for t in range(model.T)]


def _kernels_time_class(model, theta, rates):
    inv, K = class_map(model)
    lam = np.exp(np.asarray(theta).reshape(model.T, K))
    return [build_kernel(model, lam[t][inv], None, rates=rates)
            for t in range(model.T)]


def _n_classes(model):
    return class_map(model)[1]


RUNGS = {
    "scalar":      (_kernels_scalar,      lambda N, T, K: 1),
    "class":       (_kernels_class,       lambda N, T, K: K),
    "time_scalar": (_kernels_time_scalar, lambda N, T, K: T),
    "time_class":  (_kernels_time_class,  lambda N, T, K: K * T),
}

RUNG_ORDER = ("scalar", "class", "time_scalar", "time_class")


def _embed(theta, src, dst, N, T, K):
    """Lift a solved rung into a larger rung that contains it."""
    th = np.asarray(theta, dtype=float)
    if src == "scalar" and dst == "class":
        return np.full(K, th[0])
    if src == "scalar" and dst == "time_scalar":
        return np.full(T, th[0])
    if src == "scalar" and dst == "time_class":
        return np.full(K * T, th[0])
    if src == "class" and dst == "time_class":
        return np.tile(th, T)                      # same tilt at every step
    if src == "time_scalar" and dst == "time_class":
        return np.repeat(th, K)                    # same tilt for every class
    return None


# --------------------------------------------------------------------------
# optimisation
# --------------------------------------------------------------------------

def rung_ceiling(model, rung, extra_starts=None, n_restarts=2, seed=0,
                 maxiter=400):
    """Optimise the exact variance over one rung.

    L-BFGS-B with finite-difference gradients, not Nelder-Mead. At N*T = 64
    parameters a simplex method will not converge in any reasonable budget,
    while the objective here is smooth in log-lambda and cheap (~5 ms), so
    numerical gradients are affordable and vastly better conditioned.
    """
    build, dim_fn = RUNGS[rung]
    dim = dim_fn(model.cfg.N, model.T, _n_classes(model))
    rates = rate_matrices(model)
    rng = np.random.default_rng(seed)

    def objective(theta):
        var = exact_traj_variance_tv(model, build(model, theta, rates))
        if not np.isfinite(var) or var <= 0:
            return 1e10
        return float(np.log(var))

    starts = [np.zeros(dim)]
    if extra_starts:
        starts += [np.asarray(s, float) for s in extra_starts]
    starts += [rng.normal(0, 0.5, dim) for _ in range(n_restarts)]

    best = {"theta": None, "var": np.inf}
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 2})
        var = exact_traj_variance_tv(model, build(model, res.x, rates))
        if np.isfinite(var) and 0 < var < best["var"]:
            best = {"theta": res.x.copy(), "var": var}

    p = model.p_fail
    best.update(rung=rung, n_params=dim,
                vrf=(p * (1 - p) / best["var"] if best["var"] > 0 else np.inf))
    return best


def decompose(model, n_restarts=2, seed=0, maxiter=400, verbose=True):
    """Run all four product-form rungs, warm-started, and report the split.

    Returns the ceiling at each rung plus the marginal gain from adding time
    knowledge and from adding component knowledge. The residual to infinity is
    what only full state-dependence (a learned h_t) can recover.
    """
    K = _n_classes(model)
    solved, rows = {}, {}
    for rung in RUNG_ORDER:
        extra = [e for e in (_embed(solved[s]["theta"], s, rung,
                                    model.cfg.N, model.T, K) for s in solved)
                 if e is not None]
        r = rung_ceiling(model, rung, extra_starts=extra,
                         n_restarts=n_restarts, seed=seed, maxiter=maxiter)
        solved[rung] = r
        rows[rung] = r
        if verbose:
            print(f"  {rung:<20} params={r['n_params']:>4}  "
                  f"ceiling VRF = {r['vrf']:>9.4f}")

    vrf = {k: rows[k]["vrf"] for k in RUNG_ORDER}
    monotone = all(vrf[b] >= vrf[a] - 1e-9
                   for a, b in zip(RUNG_ORDER, RUNG_ORDER[1:]))

    out = {
        "rungs": rows,
        "vrf": vrf,
        "monotone": monotone,
        # marginal value of each kind of knowledge, over the scalar baseline
        "gain_component_only": vrf["class"] / vrf["scalar"],
        "gain_time_only": vrf["time_scalar"] / vrf["scalar"],
        "gain_both": vrf["time_class"] / vrf["scalar"],
        "n_classes": K,
    }
    if verbose:
        print(f"  monotone: {monotone}   (classes: {K})")
        print(f"  component knowledge alone: x{out['gain_component_only']:.3f}")
        print(f"  time knowledge alone:      x{out['gain_time_only']:.3f}")
        print(f"  both:                      x{out['gain_both']:.3f}")
        print(f"  residual to zero variance: product-form caps at "
              f"{vrf['time_class']:.2f}x, optimum is infinite")
    return out