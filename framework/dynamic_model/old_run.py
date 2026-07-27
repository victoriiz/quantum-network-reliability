"""
dynamic_model/run.py -- runner for the dynamic (mission-time) model.

KEY DIFFERENCE from the static runner: the audit samples TRAJECTORIES, not
single states. A proposal here is a rule for tilting each time-step's
transition; we roll the chain forward under it and weight each whole path
by the product of true/proposal transition ratios. Everything else (the
validate gate, the p_T ground truth, reading bias before VRF) is the same.

The zero-variance proposal is the h-transform (model.tilted_transition).
Auditing IT is the correctness check: it should give ~zero variance.
"""
import numpy as np
from framework.dynamic_model.model import DynamicConfig, DynamicModel


def audit_trajectories(model, proposal, n_paths=50_000, n_trials=10, seed=0):
    """Importance sampling over PATHS.
    `proposal(x_idx, steps_left)` returns the tilted next-state distribution
    (or None to use the true dynamics = naive).
    Weight of a path = product over steps of  P_true(x->y) / P_prop(x->y).
    """
    P = model.Pmat
    ests = np.empty(n_trials)
    ess_frac = np.empty(n_trials)
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        ws = np.empty(n_paths)
        for k in range(n_paths):
            x = model.start_idx
            logw = 0.0
            #failed = int(model.in_F[x])
            if model.in_F[x]:
                ws[k] = 1.0
                continue
            failed = 0

            for step in range(model.T):
                steps_left = model.T - step
                q = proposal(x, steps_left) if proposal else P[x]
                if q is None:
                    q = P[x]
                y = rng.choice(model.n_states, p=q)
                logw += np.log(P[x, y] + 1e-300) - np.log(q[y] + 1e-300)
                x = y
                if model.in_F[x]:
                    failed = 1
                    break
            ws[k] = np.exp(logw) * failed
        ests[t] = ws.mean()
        s1, s2 = ws.sum(), (ws ** 2).sum()
        ess_frac[t] = (s1 * s1 / s2) / n_paths if s2 > 0 else 0.0
    mean = ests.mean()
    var_single = ests.var(ddof=1) * n_paths
    pf = model.p_fail
    vrf = pf * (1 - pf) / var_single if var_single > 0 else np.inf
    bias = (mean - pf) / pf
    return {"mean": mean, "bias": bias, "vrf": vrf, "ess": ess_frac.mean()}


def run(model, verbose=True):
    v = model.validate()
    if verbose:
        print(f"[{v['name']}]  p_T = {v['p_fail']:.3e}   "
              f"failure states = {v['n_fail_states']}/{v['n_states']}")
        print(f"  rare? {v['is_rare']}   rows sum to 1? {v['rows_sum_to_1']}"
              f"   DP matches sim? {v['dp_matches_sim']} (sim={v['sim']:.3e})")
    assert v["rows_sum_to_1"], "transition rows don't sum to 1 -- Layer-1 bug"
    assert v["dp_matches_sim"], "DP disagrees with naive sim -- Layer-1 bug"

    # naive baseline (no tilting)
    naive = audit_trajectories(model, proposal=None)
    # the h-transform (zero-variance ideal) -- should be ~perfect
    h_star = audit_trajectories(model, proposal=model.tilted_transition)

    if verbose:
        print(f"\n{'proposal':<24}{'mean':>12}{'bias':>9}{'VRF':>14}{'ESS':>7}")
        print(f"{'naive (true dynamics)':<24}{naive['mean']:>12.3e}"
              f"{naive['bias']*100:>+8.2f}%{naive['vrf']:>13.1f}x{naive['ess']:>7.2f}")
        print(f"{'h-transform (ideal)':<24}{h_star['mean']:>12.3e}"
              f"{h_star['bias']*100:>+8.2f}%{h_star['vrf']:>13.1e}x{h_star['ess']:>7.2f}")
        print(f"\n  the h-transform should show ~zero variance (huge VRF, "
              f"ESS ~ 1). If it doesn't, Layer 2 has a bug.")
    return naive, h_star


def demo_instance():
    """Heterogeneous capacities (the hardness dial) + rare-ish failure."""
    N = 4
    return DynamicModel(DynamicConfig(
        N=N, T=8,
        c=np.array([3.0, 1.0, 1.0, 1.0]), c_min=2.0,
        a0=np.full(N, 0.01), gamma=np.full(N, 0.3),
        b0=np.full(N, 0.2), eta=np.full(N, 0.5),
        name="dyn-4-het"))


if __name__ == "__main__":
    run(demo_instance())