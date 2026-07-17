"""
Variational quantum proposal for latent subsystem states.
Uses PennyLane default.qubit for exact statevector simulation (m <= 16).
"""
import numpy as np
import pennylane as qml


class QuantumProposal:
    """
    Hardware-efficient ansatz (RY + ring CNOT) on m qubits.
    Prepares |psi(theta)>; proposal is q_theta(z) = |<z|psi>|^2.
    """
    def __init__(self, m, n_layers=4, seed=42):
        self.m = m
        self.n_layers = n_layers
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # One RY rotation per qubit per layer, plus initial layer
        self.n_params = m * (n_layers + 1)
        self.dev = qml.device("default.qubit", wires=m)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta):
            # Initial layer
            for i in range(m):
                qml.RY(theta[i], wires=i)
            # Entangling layers
            idx = m
            for _ in range(n_layers):
                # Ring of CNOTs
                for i in range(m):
                    qml.CNOT(wires=[i, (i + 1) % m])
                # RY rotations
                for i in range(m):
                    qml.RY(theta[idx + i], wires=i)
                idx += m
            return qml.probs()

        self.circuit = circuit

    def get_probs(self, theta):
        """Return full probability distribution over 2^m states."""
        return self.circuit(theta)

    def sample(self, theta, n_samples, rng=None):
        """Sample bitstrings from q_theta."""
        if rng is None:
            rng = np.random.default_rng()
        probs = self.get_probs(theta)
        states = np.arange(2**self.m)
        idx = rng.choice(states, size=n_samples, p=probs)
        # Convert to bitstrings
        samples = np.array([list(np.binary_repr(i, width=self.m)) for i in idx], dtype=int)
        return samples

    def log_prob(self, theta, z):
        """
        Compute log q_theta(z) for given bitstrings.

        Parameters
        ----------
        theta : np.ndarray
        z : np.ndarray, shape (batch, m)

        Returns
        -------
        log_probs : np.ndarray, shape (batch,)
        """
        probs = self.get_probs(theta)
        z = np.atleast_2d(z)
        idx = np.sum(z * (2 ** np.arange(self.m - 1, -1, -1)), axis=1)
        return np.log(probs[idx] + 1e-16)

    def init_theta(self):
        """Random initialization."""
        return self.rng.random(self.n_params) * 2 * np.pi
