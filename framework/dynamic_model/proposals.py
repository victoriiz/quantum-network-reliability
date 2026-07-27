"""
Proposal families for the mission-time model.
Goes in: framework/dynamic_model/proposals.py

A proposal is a row-stochastic kernel Q with the SAME SUPPORT as model.Pmat.
Every family here is an *odds tilt*: it multiplies the odds of each
per-component event by a positive factor. For any finite, strictly positive
tilt the support is preserved exactly, so the IS estimator stays unbiased and
the second-moment recursion never divides by zero.

The expressiveness ladder (this is the SRC result):

    family            params   depends on
    ----------------------------------------------------------
    scalar              1      nothing
    per_component       N      which component
    per_comp_repair    2N      which component, failure vs repair
    h-transform         --     state AND time  (the true optimum)

The first three are time- and state-independent tilts of the *rates*; the
underlying rates a_i(x), b_i(x) are still state-dependent because the model
makes them so. The point of the ladder is that none of these families can
express the state-and-time dependence that q*_t(x->y) ~ P(x,y) h_{t-1}(y) has.
"""

import numpy as np

# --------------------------------------------------------------------------
# rate matrices
# --------------------------------------------------------------------------

def rate_matrices(model):
    """A[x, i] = a_i(x)  and  B[x, i] = b_i(x)  for every state index x.

    Uses model._a_i / model._b_i, which are private in model.py. If you would
    rather not reach into privates, promote them to public methods there; this
    is the only coupling point.
    """
    N = model.cfg.N
    A = np.empty((model.n_states, N), dtype=float)
    B = np.empty((model.n_states, N), dtype=float)
    for xi in range(model.n_states):
        x = model.states[xi]
        A[xi] = model._a_i(x)
        B[xi] = model._b_i(x)
    return A, B


# --------------------------------------------------------------------------
# the tilt
# --------------------------------------------------------------------------

def odds_tilt(p, lam):
    """Multiply the ODDS of an event by lam.

    odds(p) = p / (1 - p);  odds(p') = lam * odds(p)
      =>  p' = lam p / (lam p + 1 - p)

    lam = 1 leaves p unchanged. lam > 1 makes the event more likely.
    Unlike a raw multiplicative tilt (p' = lam * p) this can never exceed 1,
    so no clipping is needed and the family is smooth in log(lam).
    """
    return lam * p / (lam * p + (1.0 - p))


def build_kernel(model, lam_fail, lam_repair=None, rates=None):
    """Build the (n_states, n_states) proposal kernel Q.

    lam_fail   : (N,) odds multipliers on the per-component FAILURE prob a_i(x)
    lam_repair : (N,) odds multipliers on the per-component REPAIR prob b_i(x).
                 None means 1.0 (repair untouched).
    rates      : optional (A, B) from rate_matrices(model), passed in to avoid
                 recomputing inside an optimiser loop.
    """
    N = model.cfg.N
    lam_fail = np.asarray(lam_fail, dtype=float)
    lam_repair = (np.ones(N) if lam_repair is None
                  else np.asarray(lam_repair, dtype=float))

    A, B = rate_matrices(model) if rates is None else rates
    At = odds_tilt(A, lam_fail[None, :])          # (n_states, N)
    Bt = odds_tilt(B, lam_repair[None, :])

    S = model.states                              # (n_states, N) targets
    Q = np.empty((model.n_states, model.n_states), dtype=float)

    # loop over SOURCE states, vectorise over targets: memory-safe for larger N
    for xi in range(model.n_states):
        x = S[xi]                                 # (N,)
        a = At[xi]                                # (N,)
        b = Bt[xi]
        # per-component transition prob for every target state at once
        up = np.where(S == 1, 1.0 - a, a)         # source component operational
        dn = np.where(S == 1, b, 1.0 - b)         # source component failed
        per = np.where(x[None, :] == 1, up, dn)   # (n_states, N)
        Q[xi] = per.prod(axis=1)

    return Q


# --------------------------------------------------------------------------
# family definitions: theta (unconstrained, log-space) -> (lam_fail, lam_repair)
# --------------------------------------------------------------------------

def _scalar(theta, N):
    return np.full(N, np.exp(theta[0])), np.ones(N)


def _per_component(theta, N):
    return np.exp(theta[:N]), np.ones(N)


def _per_comp_repair(theta, N):
    return np.exp(theta[:N]), np.exp(theta[N:2 * N])


FAMILIES = {
    "scalar":          (_scalar,          lambda N: 1),
    "per_component":   (_per_component,   lambda N: N),
    "per_comp_repair": (_per_comp_repair, lambda N: 2 * N),
}


def kernel_from_theta(model, family, theta, rates=None):
    unpack, _ = FAMILIES[family]
    lam_f, lam_r = unpack(np.asarray(theta, dtype=float), model.cfg.N)
    return build_kernel(model, lam_f, lam_r, rates=rates)


# --------------------------------------------------------------------------
# the optimum, for reference: Doob h-transform (state- AND time-dependent)
# --------------------------------------------------------------------------

def h_transform_sequence(model):
    """Q_seq[t] = the ideal kernel used at step t (t = 0 .. T-1).

    At step t there are steps_left = T - t transitions remaining, so
        q*_t(x -> y)  ~  P(x, y) h_{steps_left - 1}(y).

    Every trajectory that hits F carries weight exactly p_fail, and under q*
    every trajectory hits F. Variance is exactly zero. This is the audit's
    correctness gate, never an estimator you would deploy.
    """
    Q_seq = []
    for t in range(model.T):
        steps_left = model.T - t
        Q = np.empty_like(model.Pmat)
        for xi in range(model.n_states):
            row = model.tilted_transition(xi, steps_left)
            Q[xi] = model.Pmat[xi] if row is None else row
        Q_seq.append(Q)
    return Q_seq