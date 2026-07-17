# Quantum Algorithm for Network Reliability

Fault-tree reliability model with latent shared subsystems (power, cooling, switches).
GPU failures are correlated through shared infrastructure.
Classical tilting misses these correlations because it assumes product-form proposals.

## Files

| File | Purpose |
|------|---------|
| `toy_model.py` | An empirical implementation and benchmarking suite for the paper *"A Quantum Algorithm for Network Reliability"* (Pabst & Nam, arXiv:2203.10201). |
| `model.py` | `LatentNetworkModel`: exact enumeration, GPU health, connectivity, failure event |
| `classical.py` | `NaiveEstimator`, `TiltedEstimator` with scalar grid search |
| `quantum.py` | `QuantumProposal`: PennyLane HEA ansatz (RY + ring CNOT) |
| `train_exact.py` | Exact-statevector forward-KL training + evaluation |
| `metrics.py` | VRF, ESS, bias computation |
| `run_validation.py` | End-to-end validation on 14-subsystem, 16-GPU instance |

## Validation Instance

- 4 nodes × 4 GPUs = 16 GPUs
- 8 node-level subsystems (PSU + Cooler per node)
- 4 inter-node switches (ring topology)
- 2 rack-level subsystems (PDU for nodes 0–1, CDU for nodes 2–3)
- **m = 14 subsystems** → exact enumeration feasible ($2^{14}$ = 16,384 states)
- Job requests 8 GPUs (nodes 0 and 1), tolerates up to 2 losses, requires connectivity

## Running

```bash
pip install pennylane numpy
python run_validation.py
```

## Scaling to m = 26 (Benchmark Mode)

For the 8-node benchmark (m = 26), exact statevector is too expensive.
Switch to REINFORCE/sampled gradients:
1. Replace `train_exact.py` with a REINFORCE loop using `proposal.sample()`.
2. Use reverse-KL (maximum likelihood on samples from $p^*$) instead of exact forward-KL.
3. Compute baseline from running average of log-probabilities.


## Relation to Prior Work

This generalizes `qnet/` (Phase 4.5 in quantum-sampling project), which used i.i.d. edge failures.
Here, failures are induced by latent subsystem states, making the problem #P-complete
and classically hard for product-form tilting.
