"""
Exact trajectory-space variance and family ceilings.
Goes in: framework/dynamic_model/ceiling.py

Replaces exact_traj_ceiling()'s brute-force param_grid with an optimiser, so
families with N or 2N parameters are reachable. The grid was fine for a
one-knob scalar family; it is not fine in 8 dimensions.

What a "ceiling" is: the best single-sample variance ANY member of a proposal
family can achieve, computed exactly, with no sampling. It is a measuring
instrument, not an estimator. It never returns an estimate of p_fail.
"""

import numpy as np
from scipy.optimize import minimize

from proposals import FAMILIES, kernel_from_theta, rate_matrices


# --------------------------------------------------------------------------
# exact single-sample variance of the mission-time IS estimator
# --------------------------------------------------------------------------

def exact_traj_variance(model, Q, check_support=True):
    """Second-moment backward recursion, stopping at first entry to F.

        R(x, y) = P(x, y)^2 / Q(x, y)
        M_0(x)  = 0
        M_s(x)  = sum_{y in F} R(x, y) + sum_{y not in F} R(x, y) M_{s-1}(y)
        Var     = M_T(start) - p_fail^2

    Derivation of one step, from x not in F: draw y ~ Q and carry the weight
    P(x,y)/Q(x,y). If y is in F the path stops with squared weight
    (P/Q)^2, whose Q-expectation is sum_{y in F} P^2/Q. Otherwise it continues
    and contributes (P/Q)^2 M_{s-1}(y), expectation sum_{y not in F} R M_{s-1}.

    check_support guards the failure mode the original code silently ignored:
    if Q(x,y) = 0 anywhere P(x,y) > 0, the estimator MISSES mass and is biased,
    and setting R = 0 there understates the variance instead of flagging it.
    Odds tilts never trigger this; learned or clipped proposals can.
    """
    P, in_F, T = model.Pmat, model.in_F, model.T

    if check_support and np.any((P > 0) & (Q <= 0)):
        return np.inf                      # missing support => biased, unusable

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(Q > 0, P * P / Q, 0.0)

    fail_cols = in_F == 1
    safe_cols = ~fail_cols
    hit_now = R[:, fail_cols].sum(axis=1)  # constant across the recursion

    M = np.zeros(model.n_states)
    for _ in range(T):
        M = hit_now + R[:, safe_cols] @ M[safe_cols]
        M[fail_cols] = 0.0                 # F is absorbing; never step out of it

    var = M[model.start_idx] - model.p_fail ** 2
    return float(var)


def vrf_from_var(model, var):
    """Variance reduction factor against naive Monte Carlo.

    Naive MC draws one trajectory and records the indicator of failure, whose
    single-sample variance is p(1-p). VRF = Var_naive / Var_Q.
    """
    p = model.p_fail
    if not np.isfinite(var):
        return 0.0
    if var <= 0:
        return np.inf
    return float(p * (1.0 - p) / var)


def exact_traj_vrf(model, Q, check_support=True):
    return vrf_from_var(model, exact_traj_variance(model, Q, check_support))


# --------------------------------------------------------------------------
# family ceiling by optimisation
# --------------------------------------------------------------------------

def family_ceiling(model, family, n_restarts=6, seed=0, maxiter=2000,
                   verbose=False, extra_starts=None):
    """Optimise the exact variance over a proposal family.

    Optimises in theta = log(lambda) so lambda > 0 is automatic and the
    objective is smooth and unconstrained; minimises log(variance) for
    numerical stability across many orders of magnitude.

    Nelder-Mead is local, so we restart: theta = 0 (naive MC, lambda = 1) plus
    random restarts. Without restarts the per-component family can park in a
    local optimum and understate its own ceiling, which would silently fake
    the headline result.
    """
    unpack, dim_fn = FAMILIES[family]
    dim = dim_fn(model.cfg.N)
    rates = rate_matrices(model)           # compute once, reuse every call
    rng = np.random.default_rng(seed)

    def objective(theta):
        Q = kernel_from_theta(model, family, theta, rates=rates)
        var = exact_traj_variance(model, Q)
        if not np.isfinite(var) or var <= 0:
            return 1e10
        return float(np.log(var))

    starts = [np.zeros(dim)]
    if extra_starts:                       # warm starts from smaller families
        starts += [np.asarray(s, dtype=float) for s in extra_starts]
    starts += [rng.normal(0.0, 1.0, size=dim) for _ in range(n_restarts - 1)]

    best = {"theta": None, "var": np.inf, "vrf": 0.0, "n_params": dim}
    for k, x0 in enumerate(starts):
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "maxfev": maxiter,
                                "xatol": 1e-6, "fatol": 1e-10,
                                "adaptive": True})
        Q = kernel_from_theta(model, family, res.x, rates=rates)
        var = exact_traj_variance(model, Q)
        if np.isfinite(var) and 0 < var < best["var"]:
            best = {"theta": res.x.copy(), "var": var,
                    "vrf": vrf_from_var(model, var), "n_params": dim}
        if verbose:
            print(f"    restart {k}: log-var {objective(res.x):+.4f}  "
                  f"vrf {vrf_from_var(model, var):.3f}")

    best["family"] = family
    best["lam_fail"], best["lam_repair"] = unpack(best["theta"], model.cfg.N)
    return best


# --------------------------------------------------------------------------
# the ladder, with the two diagnostics that catch a failed optimiser
# --------------------------------------------------------------------------

LADDER = ("scalar", "per_component", "per_comp_repair")


def _embed(theta, src, dst, N):
    """Lift a solution from a smaller family into a larger one that contains it.

    scalar -> per_component     : broadcast the single tilt to all components
    per_component -> per_comp_repair : append zeros (log 1 = repair untouched)

    This is what makes the ladder monotone by construction: the larger family's
    optimiser starts from a point that already achieves the smaller family's
    ceiling, so it can only improve.
    """
    theta = np.asarray(theta, dtype=float)
    if src == "scalar" and dst == "per_component":
        return np.full(N, theta[0])
    if src == "scalar" and dst == "per_comp_repair":
        return np.concatenate([np.full(N, theta[0]), np.zeros(N)])
    if src == "per_component" and dst == "per_comp_repair":
        return np.concatenate([theta, np.zeros(N)])
    return None


def symmetry_classes(model):
    """Group components that are exchangeable under the model parameters.

    Two components with identical (c_i, a0_i, b0_i, gamma_i, eta_i) are
    interchangeable, so ANY optimum must assign them the same tilt. If the
    optimiser returns different tilts within a class, it has not converged.
    This diagnostic is free and it is how the 16-parameter family was caught
    failing while still looking plausible.
    """
    cfg = model.cfg
    key = np.stack([cfg.c, cfg.a0, cfg.b0, cfg.gamma, cfg.eta], axis=1)
    _, inv = np.unique(np.round(key, 12), axis=0, return_inverse=True)
    return inv


def symmetry_violation(model, lam):
    """Max within-class spread of the tilt vector, relative to the class mean."""
    inv = symmetry_classes(model)
    lam = np.asarray(lam, dtype=float)
    worst = 0.0
    for cls in np.unique(inv):
        v = lam[inv == cls]
        if v.size > 1 and v.mean() != 0:
            worst = max(worst, float((v.max() - v.min()) / abs(v.mean())))
    return worst


def run_ladder(model, families=LADDER, n_restarts=4, seed=0, maxiter=2000,
               sym_tol=1e-3, verbose=True):
    """Optimise each family, warm-starting from every smaller one already solved.

    Enforces and reports two invariants a correct run must satisfy:

      1. MONOTONICITY. The families are nested (scalar subset per_component
         subset per_comp_repair), so ceilings must be non-decreasing. A drop
         means the optimiser failed, never that the bigger family is worse.
      2. SYMMETRY. Exchangeable components must receive equal tilts.

    Either violation makes the number unreportable. Do not paper over it by
    picking the best of several runs.
    """
    solved, results = {}, {}
    for fam in families:
        extra = [e for e in (_embed(solved[s]["theta"], s, fam, model.cfg.N)
                             for s in solved) if e is not None]
        res = family_ceiling(model, fam, n_restarts=n_restarts, seed=seed,
                             maxiter=maxiter, extra_starts=extra)
        res["sym_violation"] = symmetry_violation(model, res["lam_fail"])
        res["sym_ok"] = res["sym_violation"] < sym_tol
        solved[fam] = res
        results[fam] = res
        if verbose:
            print(f"  {fam:<17} params={res['n_params']:>3}  "
                  f"ceiling VRF={res['vrf']:>9.4f}  "
                  f"sym spread={res['sym_violation']:.2e} "
                  f"{'ok' if res['sym_ok'] else 'FAIL'}")

    order = [results[f]["vrf"] for f in families]
    monotone = all(b >= a - 1e-9 for a, b in zip(order, order[1:]))
    if verbose and not monotone:
        print("  !! MONOTONICITY VIOLATED: a nested family reported a lower "
              "ceiling. The optimiser did not converge. Do not report these.")
    return {"families": results, "monotone": monotone,
            "all_symmetric": all(results[f]["sym_ok"] for f in families)}


# --------------------------------------------------------------------------
# verification: brute-force the second moment by enumerating stopped paths
# --------------------------------------------------------------------------

def brute_force_variance(model, Q):
    """Enumerate every stopped trajectory and accumulate E_Q[W^2 1_hit].

    Exponential in T, so only usable on tiny instances. This is the check that
    the recursion above is correct; run it once on N=3, T=3 and never again.
    """
    P, in_F, T = model.Pmat, model.in_F, model.T
    total = 0.0

    def walk(x, w2, q_prob, steps_left):
        nonlocal total
        if steps_left == 0:
            return
        for y in range(model.n_states):
            qxy = Q[x, y]
            if qxy <= 0:
                continue
            ratio2 = (P[x, y] / qxy) ** 2
            if in_F[y] == 1:
                total += q_prob * qxy * w2 * ratio2
            else:
                walk(y, w2 * ratio2, q_prob * qxy, steps_left - 1)

    walk(model.start_idx, 1.0, 1.0, T)
    return float(total - model.p_fail ** 2)