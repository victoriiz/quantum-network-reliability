"""
framework/static_model/quantum_assess.py

Assess a quantum circuit proposal on the static fault-tree model, against
the strongest classical opponents, on the same audit:

    naive MC  |  cross-entropy (adaptive IS)  |  exact tilt-family ceiling

The quantum proposal is a Layer-3 proposal like any other; it is scored by
the same audit (bias first, then VRF, then ESS) and judged against the
exact ceiling. Multiple seeds test stability vs luck.
"""
import numpy as np
from model import FaultTreeConfig, FaultTreeModel
import estimators as est


def rare_instance():
    """Small, calibrated-to-be-rare-AND-concentrated instance."""
    return FaultTreeModel(FaultTreeConfig(
        m=6,
        p_fail=np.array([0.04, 0.04, 0.03, 0.03, 0.02, 0.02]),
        coverage={0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}},
        n_gpu=6, job={0, 1, 2, 3, 4, 5},
        min_healthy=3, name="rare-6"))


def train_quantum_proposal(model, layers=3, steps=300, seed=0):
    """q(b) = |<b|psi(theta)>|^2 from an RY+CNOT circuit, trained by
    forward KL to p*. Returns the proposal vector, or None if PennyLane
    is unavailable."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError:
        return None
    m = model.n_components
    pstar = est.ideal_proposal(model)
    dev = qml.device("default.qubit", wires=m)

    @qml.qnode(dev)
    def circuit(theta):
        theta = theta.reshape(layers, m)
        for L in range(layers):
            for i in range(m):
                qml.RY(theta[L, i], wires=i)
            for i in range(m - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.probs(wires=range(m))

    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, layers * m), requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    ps = pnp.array(pstar, requires_grad=False)
    for _ in range(steps):
        theta = opt.step(lambda t: -pnp.sum(ps * pnp.log(circuit(t) + 1e-12)),
                         theta)
    return np.array(circuit(theta))


def assess(model, n_seeds=5, seed0=0):
    v = model.validate()
    print(f"[{v['name']}] P_fail={v['p_fail']:.3e}  "
          f"fail states {v['n_fail_states']}/{v['n_states']}  "
          f"usable={v['usable']}")
    assert v["usable"], "instance not rare/concentrated -- fix before assessing"

    # --- the strongest classical opponents ---
    ceil = est.tilt_family_ceiling(model)
    ce = est.cross_entropy(model, seed=seed0)
    print(f"\nexact tilt-family ceiling : {ceil['vrf']:9.1f}x   (theoretical best "
          f"of the product family)")
    print(f"cross-entropy (adaptive)  : {ce['vrf']:9.1f}x   bias {ce['bias']*100:+.2f}%"
          f"   (practical adaptive IS)\n")

    def audited(name, q_list):
        vrfs, biases = [], []
        for q in q_list:
            r = est.audit(model, q, n_samples=100_000, n_trials=5)
            vrfs.append(r["vrf"]); biases.append(r["bias"])
        vm, vs = np.mean(vrfs), np.std(vrfs)
        verdict = ("beats ceiling" if vm > ceil["vrf"] else
                   "~ties ceiling" if vm > 0.8 * ceil["vrf"] else "below")
        print(f"{name:<30}{vm:>11.1f} +/-{vs:>7.1f}"
              f"{np.mean(biases)*100:>+8.2f}%   {verdict}")

    print(f"{'proposal':<30}{'VRF (mean+/-sd)':>18}{'bias':>9}")
    audited("naive MC", [est.naive_proposal(model)])
    qs = [train_quantum_proposal(model, seed=s) for s in range(n_seeds)]
    if any(q is None for q in qs):
        print("\n(PennyLane not installed -- quantum row skipped; classical "
              "comparison shown)")
        return
    audited(f"QUANTUM circuit (n={n_seeds})", qs)
    print(f"\nread: bias first (must be ~0), then VRF vs ceiling and CE.")
    print(f"the +/- on the quantum row shows stability across seeds "
          f"(large sd = lucky, not good).")


if __name__ == "__main__":
    assess(rare_instance())