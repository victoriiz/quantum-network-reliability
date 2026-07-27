"""
Sampling audit for the DYNAMIC (trajectory) model.
Goes in: framework/dynamic_model/audit.py

estimators.py's audit() is static-model code: it reads model.p, model.failmask
and model.component_bits, none of which DynamicModel defines. This is the
trajectory equivalent, with the same scorecard so results are comparable.

Why this exists when ceiling.py already computes the variance exactly:
the ceiling says what is ACHIEVABLE IN THEORY; the audit says what your code
ACTUALLY DOES. The -11.5% h-transform bias was a provably zero-variance
proposal with a wrong stopping rule. The ceiling could not have caught it.
Only sampling did.

Bias is read FIRST. A large VRF around a biased estimate is worthless.
"""

import numpy as np


def _sample_step(rng, cum, cur):
    """Draw the next state for a batch of paths by inverse-CDF lookup.

    rng.choice(p=...) is ~50 us per call, so a 20k x T=8 audit would take
    minutes. This does the whole batch in one vectorised comparison.
    """
    u = rng.random(cur.size)
    nxt = (cum[cur] < u[:, None]).sum(axis=1)
    return np.minimum(nxt, cum.shape[1] - 1)     # guard fp round-off at the top


def audit_trajectory(model, Q=None, Q_seq=None, n_paths=20_000, n_trials=20,
                     seed=0):
    """Importance-sampling scorecard for a mission-time proposal.

    Pass EITHER Q (one time-homogeneous kernel) or Q_seq (list of T kernels,
    Q_seq[t] used at step t). The h-transform needs Q_seq because the ideal
    proposal is time-dependent.

    THE STOPPING RULE: mission failure is a HITTING event. Reweighting stops
    at first entry to F. Continuing to reweight past the hit is the bug that
    produced -11.5% bias on a proposal that is provably exact.
    """
    if (Q is None) == (Q_seq is None):
        raise ValueError("pass exactly one of Q or Q_seq")

    P, in_F, T = model.Pmat, model.in_F, model.T
    start = model.start_idx
    if in_F[start] == 1:
        raise ValueError("start state is already in F; the instance is broken")

    kernels = Q_seq if Q_seq is not None else [Q] * T
    if len(kernels) != T:
        raise ValueError(f"Q_seq must have length T={T}, got {len(kernels)}")
    cums = [K.cumsum(axis=1) for K in kernels]

    rng = np.random.default_rng(seed)
    ests = np.empty(n_trials)
    ess_frac = np.empty(n_trials)

    for trial in range(n_trials):
        cur = np.full(n_paths, start, dtype=np.int64)
        w = np.ones(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        hit_w = np.zeros(n_paths)          # 0 for paths that never reach F

        for t in range(T):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            src = cur[idx]
            nxt = _sample_step(rng, cums[t], src)

            q = kernels[t][src, nxt]
            ratio = np.where(q > 0, P[src, nxt] / np.where(q > 0, q, 1.0), 0.0)
            w[idx] *= ratio

            hit = in_F[nxt] == 1
            hit_idx = idx[hit]
            hit_w[hit_idx] = w[hit_idx]    # record and STOP: hitting event
            alive[hit_idx] = False
            cur[idx] = nxt

        ests[trial] = hit_w.mean()
        s1, s2 = hit_w.sum(), (hit_w ** 2).sum()
        ess_frac[trial] = (s1 * s1 / s2) / n_paths if s2 > 0 else 0.0

    p = model.p_fail
    mean = float(ests.mean())
    var_single = float(ests.var(ddof=1) * n_paths)
    bias = (mean - p) / p
    ci = 1.96 * float(ests.std(ddof=1)) / np.sqrt(n_trials)
    vrf = (p * (1 - p) / var_single) if var_single > 0 else np.inf
    mse = bias ** 2 * p ** 2 + var_single / n_paths

    return {"mean": mean, "ci": ci, "bias": bias, "vrf": vrf,
            "ess": float(ess_frac.mean()), "mse": mse,
            "n_paths": n_paths, "n_trials": n_trials}


def naive_audit(model, **kw):
    """Baseline: sample the true chain, no reweighting. VRF is 1 by definition."""
    return audit_trajectory(model, Q=model.Pmat.copy(), **kw)