"""
Exact-statevector training for the quantum proposal.
Minimizes forward KL(p* || q_theta) using exact enumeration.
Suitable for m <= 16.
"""
import numpy as np
import pennylane as qml
from quantum import QuantumProposal


def train_exact(model, p_star_dict, n_layers=4, lr=0.1, n_steps=200, 
                verbose=True, seed=42):
    """
    Train quantum proposal by exact forward KL minimization.

    Parameters
    ----------
    model : LatentNetworkModel
    p_star_dict : dict
        Mapping state_tuple -> unnormalized probability (from exact_pfail).
    n_layers : int
    lr : float
        Learning rate for gradient descent.
    n_steps : int
    verbose : bool
    seed : int

    Returns
    -------
    theta : np.ndarray
        Trained parameters.
    loss_history : list
    """
    rng = np.random.default_rng(seed)
    proposal = QuantumProposal(model.m, n_layers=n_layers, seed=seed)
    theta = proposal.init_theta()

    # Normalize p_star
    total = sum(p_star_dict.values())
    p_star = {k: v / total for k, v in p_star_dict.items()}

    loss_history = []

    for step in range(n_steps):
        # Compute q_theta exactly
        q_probs = proposal.get_probs(theta)

        # Forward KL = -sum p*(z) log q(z)
        loss = 0.0
        grad = np.zeros(proposal.n_params)

        for bits, p_val in p_star.items():
            idx = sum(b * (2 ** (model.m - 1 - i)) for i, b in enumerate(bits))
            q_val = q_probs[idx]
            if q_val > 1e-16:
                loss -= p_val * np.log(q_val)
                # Gradient of -log q_val w.r.t. theta
                # Use parameter-shift for the probability
                g = qml.gradients.param_shift(proposal.circuit)(theta)
                # g shape is (2^m, n_params) for probs output
                grad -= p_val * g[idx] / q_val
            else:
                loss += 1e6  # penalty for zero mass

        theta = theta - lr * grad
        loss_history.append(float(loss))

        if verbose and step % 20 == 0:
            print(f"Step {step:4d}: KL = {loss:.4f}")

    return theta, loss_history


def evaluate_quantum(model, p_star_dict, theta, n_samples=200000, n_trials=20, seed=42):
    """
    Evaluate trained quantum proposal with IS.

    Returns
    -------
    estimates : np.ndarray, shape (n_trials,)
    weights : np.ndarray, shape (n_trials, n_samples)
    """
    rng = np.random.default_rng(seed)
    proposal = QuantumProposal(model.m, seed=seed)

    # Normalize p_star to get target distribution
    total = sum(p_star_dict.values())

    estimates = []
    weights_all = []

    for t in range(n_trials):
        z = proposal.sample(theta, n_samples, rng=rng)
        log_q = proposal.log_prob(theta, z)

        # Compute p(z) * 1_fail(z) for each sample
        w = np.zeros(n_samples)
        for i, zi in enumerate(z):
            bits = tuple(zi.tolist())
            if bits in p_star_dict:
                w[i] = p_star_dict[bits] / np.exp(log_q[i])

        estimates.append(np.mean(w))
        weights_all.append(w)

    return np.array(estimates), np.array(weights_all)
