"""
static_model/run.py -- ties the static model's layers into one run.

Flow:  build model -> validate -> self-check p* ->
        compute exact ceiling -> audit every proposal -> print table.
"""
import numpy as np
from model import FaultTreeConfig, FaultTreeModel
import estimators as est


def run(model, seed=0, verbose=True):
    # --- gate 1: is the instance usable (rare AND concentrated)? ----------
    v = model.validate()
    if verbose:
        print(f"[{v['name']}]  P_fail = {v['p_fail']:.3e}   "
              f"failure states = {v['n_fail_states']}/{v['n_states']}")
        print(f"  rare? {v['is_rare']}   concentrated? {v['is_concentrated']}"
              f"   -> {'USABLE' if v['usable'] else 'FIX THE INSTANCE FIRST'}")
    if not v["usable"]:
        print("  (running anyway for illustration, but the results are not "
              "a meaningful rare-event study)")

    # --- gate 2: self-check that p* is really zero-variance ---------------
    pstar = est.ideal_proposal(model)
    chk = est.audit(model, pstar, n_samples=50_000, n_trials=5, seed=seed)
    assert abs(chk["bias"]) < 0.05 and chk["vrf"] > 1e6, \
        f"p* self-check FAILED (bias {chk['bias']:.3f}, VRF {chk['vrf']:.1e}) "\
        f"-> Layer 1 or 2 has a bug"
    if verbose:
        print(f"  p* self-check OK (VRF {chk['vrf']:.1e}, "
              f"bias {chk['bias']*100:+.2f}%)\n")

    # --- the honest opponent: exact tilt-family ceiling ------------------
    ceil = est.tilt_family_ceiling(model, seed=seed)

    # --- audit every proposal --------------------------------------------
    rows = [
        ("naive MC", est.audit(model, est.naive_proposal(model), seed=seed),
         False),
        ("scalar tilt (0.3)",
         est.audit(model, est.tilt_proposal(model, np.full(model.n_components,
                   0.3)), seed=seed), False),
        ("per-component tilt (ceiling)",
         est.audit(model, est.tilt_proposal(model, ceil["p_tilt"]), seed=seed),
         True),
    ]
    if verbose:
        print(f"{'method':<30}{'mean':>12}{'bias':>9}{'MSE':>11}"
              f"{'VRF':>12}{'ESS':>7}")
        for name, r, is_ceil in rows:
            star = "  <- exact family ceiling" if is_ceil else ""
            print(f"{name:<30}{r['mean']:>12.3e}{r['bias']*100:>+8.2f}%"
                  f"{r['mse']:>11.1e}{r['vrf']:>11.1f}x{r['ess']:>7.2f}{star}")
        print(f"\n  exact ceiling VRF (no sampling): {ceil['vrf']:.1f}x  "
              f"at tilt {np.round(ceil['p_tilt'], 3)}")
        print(f"  -> any TRAINED proposal must beat {ceil['vrf']:.0f}x to matter.")
    return rows, ceil


def demo_instance():
    """A small, calibrated-to-be-RARE instance. Edit to change the study."""
    return FaultTreeModel(FaultTreeConfig(
        m=6,
        p_fail=np.array([0.04, 0.04, 0.03, 0.03, 0.02, 0.02]),
        coverage={0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}},
        n_gpu=6, job={0, 1, 2, 3, 4, 5},
        min_healthy=3,        # tolerate 3 losses -> rare AND concentrated
        name="rare-6"))


if __name__ == "__main__":
    run(demo_instance())