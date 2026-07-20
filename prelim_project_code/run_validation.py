#!/usr/bin/env python3
"""
End-to-end validation script for the latent network reliability model.
Config: 4 nodes, 4 GPUs each, 14 subsystems (8 node-level + 4 inter-node + 2 rack-level).
Exact enumeration for ground truth; exact-statevector quantum training.
"""
import numpy as np
from model import LatentNetworkModel
from classical import NaiveEstimator, TiltedEstimator
from quantum import QuantumProposal
from train_exact import train_exact, evaluate_quantum
from metrics import compute_metrics


def build_validation_model():
    """
    Build the 14-subsystem validation instance.

    Topology:
      - 4 nodes, 4 GPUs per node => 16 GPUs total
      - Node i has GPUs [4*i, 4*i+1, 4*i+2, 4*i+3]
      - Subsystems:
        * Node-level: PSU_i, Cooler_i  (8 subsystems, s=0..7)
        * Inter-node: Switch_{i,(i+1)%4} for ring (4 subsystems, s=8..11)
        * Rack-level: PDU_A (nodes 0,1), CDU_B (nodes 2,3) (2 subsystems, s=12,13)
    """
    n_nodes = 4
    gpus_per_node = 4
    n_gpus = n_nodes * gpus_per_node

    # Build coverage map
    coverage = {}
    sid = 0

    # Node-level PSUs and Coolers
    for i in range(n_nodes):
        gpus = set(range(4*i, 4*i + 4))
        coverage[sid] = gpus  # PSU_i
        sid += 1
        coverage[sid] = gpus  # Cooler_i
        sid += 1

    # Inter-node switches (ring: 0-1, 1-2, 2-3, 3-0)
    # These don't kill GPUs directly, but they affect connectivity.
    # For simplicity in this validation, we model them as covering the GPUs 
    # they connect (if switch fails, GPUs on both ends lose external connectivity).
    # To keep the failure event non-trivial, we treat them as GPU-killing for now
    # (a failed switch kills the 2 GPUs it bridges). This is a modeling choice.
    ring_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for a, b in ring_pairs:
        gpus = {4*a, 4*a+1, 4*b, 4*b+1}  # one GPU from each node as "bridge"
        coverage[sid] = gpus
        sid += 1

    # Rack-level: PDU_A powers nodes 0,1; CDU_B cools nodes 2,3
    coverage[sid] = set(range(0, 8))   # PDU_A
    sid += 1
    coverage[sid] = set(range(8, 16))  # CDU_B
    sid += 1

    m = sid
    assert m == 14, f"Expected 14 subsystems, got {m}"

    # Prior failure probabilities (small)
    p_fail = np.full(m, 0.01)
    # Rack subsystems are more reliable
    p_fail[-2:] = 0.005
    # Inter-node switches slightly less reliable
    p_fail[8:12] = 0.015

    # Connectivity: full mesh within node; ring between nodes
    edges = []
    for i in range(n_nodes):
        base = 4*i
        for u in range(base, base+4):
            for v in range(u+1, base+4):
                edges.append((u, v))
    # Inter-node ring edges (connect one GPU from each node)
    for a, b in ring_pairs:
        edges.append((4*a, 4*b))

    # Job: requests 8 GPUs (all of nodes 0 and 1)
    job_gpus = set(range(0, 8))
    min_healthy = 6  # tolerate up to 2 GPU losses

    model = LatentNetworkModel(
        n_gpus=n_gpus,
        subsystem_coverage=coverage,
        p_fail=p_fail,
        connectivity_edges=edges,
        job_gpus=job_gpus,
        min_healthy=min_healthy,
        require_connected=True
    )
    return model


def main():
    print("=" * 60)
    print("Latent Network Reliability — Validation Run (m=14)")
    print("=" * 60)

    model = build_validation_model()
    print(f"Model: {model.n_gpus} GPUs, {model.m} subsystems")
    print(f"Job: {len(model.job_gpus)} GPUs, min_healthy={model.min_healthy}")

    # 1. Exact ground truth
    print("[1] Computing exact ground truth by enumeration...")
    p_exact, p_star_dict, fail_states = model.exact_pfail()
    print(f"    Exact P_fail = {p_exact:.6e}")
    print(f"    Number of failure states = {len(fail_states)} / {2**model.m}")

    # 2. Naive Monte Carlo
    print("[2] Naive Monte Carlo...")
    naive = NaiveEstimator(model)
    n_samples = 200_000
    n_trials = 20
    rng = np.random.default_rng(42)
    naive_weights = []
    for t in range(n_trials):
        _, _, w, _ = naive.estimate(n_samples, rng=rng)
        naive_weights.append(w)
    naive_weights = np.array(naive_weights)
    m_naive = compute_metrics(naive_weights, p_exact, n_samples)
    print(f"    Mean estimate: {m_naive['mean_estimate']:.6e}")
    print(f"    Rel. bias: {m_naive['rel_bias']*100:.2f}%")
    print(f"    VRF: {m_naive['VRF']:.1f}x")
    print(f"    ESS: {m_naive['ESS']:.0f}")

    # 3. Classical tilting (grid search over scalar tilt)
    print("[3] Classical per-subsystem tilting (scalar grid search)...")
    p_grid = np.linspace(0.02, 0.20, 20)
    best_p, best_vrf, best_est = TiltedEstimator.grid_search(
        model, n_samples=50_000, p_grid=p_grid, rng=np.random.default_rng(42)
    )
    print(f"    Best tilt p' = {best_p:.4f}, VRF = {best_vrf:.1f}x")

    # Evaluate best tilt with full budget
    tilt_weights = []
    for t in range(n_trials):
        _, _, w, _ = best_est.estimate(n_samples, rng=np.random.default_rng(42+t))
        tilt_weights.append(w)
    tilt_weights = np.array(tilt_weights)
    m_tilt = compute_metrics(tilt_weights, p_exact, n_samples)
    print(f"    Full-run mean: {m_tilt['mean_estimate']:.6e}")
    print(f"    Rel. bias: {m_tilt['rel_bias']*100:.2f}%")
    print(f"    VRF: {m_tilt['VRF']:.1f}x")
    print(f"    ESS: {m_tilt['ESS']:.0f}")

    # 4. Quantum IS (exact statevector)
    print("[4] Quantum IS (exact statevector, forward KL)...")
    print("    Training...")
    theta, loss_hist = train_exact(
        model, p_star_dict, n_layers=4, lr=0.15, n_steps=300, 
        verbose=True, seed=42
    )
    print(f"    Final KL = {loss_hist[-1]:.4f} nats")

    print("    Evaluating...")
    estimates, q_weights = evaluate_quantum(
        model, p_star_dict, theta, n_samples=n_samples, n_trials=n_trials, seed=42
    )
    m_quantum = compute_metrics(q_weights, p_exact, n_samples)
    print(f"    Mean estimate: {m_quantum['mean_estimate']:.6e}")
    print(f"    Rel. bias: {m_quantum['rel_bias']*100:.2f}%")
    print(f"    VRF: {m_quantum['VRF']:.1f}x")
    print(f"    ESS: {m_quantum['ESS']:.0f}")

    # 5. Summary table
    print("" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Bias':>10} {'VRF':>10} {'ESS':>10}")
    print("-" * 60)
    print(f"{'Naive MC':<25} {m_naive['rel_bias']*100:>9.2f}% {m_naive['VRF']:>10.1f}x {m_naive['ESS']:>10.0f}")
    print(f"{f'Tilt (p={best_p:.3f})':<25} {m_tilt['rel_bias']*100:>9.2f}% {m_tilt['VRF']:>10.1f}x {m_tilt['ESS']:>10.0f}")
    print(f"{'Quantum IS (KL)':<25} {m_quantum['rel_bias']*100:>9.2f}% {m_quantum['VRF']:>10.1f}x {m_quantum['ESS']:>10.0f}")
    print(f"Exact P_fail = {p_exact:.6e}")


if __name__ == "__main__":
    main()
