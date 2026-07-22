"""
quantum_assess.py -- plug a QUANTUM proposal into the static pipeline and
assess it: same audit, same exact ceiling, multiple seeds.

The quantum circuit is a Layer-3 proposal like any other. This file is the
experiment that fills the "???" row of the comparison table.

Runs on the static FaultTreeModel (its measurement distribution over 2^m
states IS a proposal). Uses a small model so training is fast enough to
iterate. Requires pennylane; falls back to a classical "trained softmax"
stand-in if pennylane is absent, so you can see the ASSESSMENT LOGIC even
without the quantum backend.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from model import FaultTreeConfig, FaultTreeModel
import estimators as est


def rare_instance():
    """A small, ACTUALLY-RARE instance (validated below)."""
    return FaultTreeModel(FaultTreeConfig(
        m=6,
        p_fail=np.array([0.04, 0.04, 0.03, 0.03, 0.02, 0.02]),
        coverage={0:{0},1:{1},2:{2},3:{3},4:{4},5:{5}},
        n_gpu=6, job={0, 1, 2, 3, 4, 5},
        min_healthy=3,
        name="rare-6"))


# ---------------------------------------------------------------------
# THE QUANTUM PROPOSAL.  q(b) = |<b|psi(theta)>|^2 from a trained circuit.
# Trained by forward-KL to p* -- identical objective to the classical
# pipeline, so the only thing being tested is the proposal family.
# ---------------------------------------------------------------------
def train_quantum_proposal(model, layers=3, steps=300, seed=0):
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


# ---------------------------------------------------------------------
# THE ASSESSMENT: multi-seed comparison vs the ceiling.
# ---------------------------------------------------------------------
def assess(model, n_seeds=5):
    v = model.validate()
    print(f"[{v['name']}] P_fail={v['p_fail']:.3e}  "
          f"fail states {v['n_fail_states']}/{v['n_states']}  "
          f"usable={v['usable']}\n")
    assert v["usable"], "instance not rare/concentrated -- fix before assessing"

    ceil = est.tilt_family_ceiling(model)
    print(f"exact tilt-family ceiling: {ceil['vrf']:.1f}x  "
          f"(the opponent to beat)\n")

    # classical reference rows
    print(f"{'proposal':<26}{'VRF (mean+/-sd)':>22}{'bias':>9}{'vs ceiling':>13}")
    def line(name, q_list):
        vrfs, biases = [], []
        for q in q_list:
            r = est.audit(model, q, n_samples=100_000, n_trials=5)
            vrfs.append(r["vrf"]); biases.append(r["bias"])
        vm, vs = np.mean(vrfs), np.std(vrfs)
        verdict = ("BEATS" if vm > ceil["vrf"] else
                   "~ties" if vm > 0.8 * ceil["vrf"] else "below")
        print(f"{name:<26}{vm:>13.1f} +/-{vs:>5.1f}"
              f"{np.mean(biases)*100:>+8.2f}%{verdict:>13}")

    line("naive MC", [est.naive_proposal(model)])
    line("per-component tilt", [est.tilt_proposal(model, ceil["p_tilt"])])

    qs = []
    for s in range(n_seeds):
        q = train_quantum_proposal(model, seed=s)
        if q is None:
            print("\n(pennylane not installed -- quantum row skipped; "
                  "assessment logic shown for classical rows)")
            return
        qs.append(q)
    line(f"QUANTUM circuit (n={n_seeds})", qs)
    print(f"\nread: bias first (all must be ~0), then VRF vs the ceiling.")
    print(f"the +/- on the quantum row tells you if it's STABLE or LUCKY.")


if __name__ == "__main__":
    assess(rare_instance())