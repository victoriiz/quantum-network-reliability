"""
Latent subsystem network reliability model.
Subsystems (power, cooling, switches) fail independently with small probability.
GPU health is a deterministic function of subsystem states.
Job failure is a topological/graph condition on the surviving GPUs.
"""
import numpy as np
from itertools import product
from collections import deque


class LatentNetworkModel:
    """
    Fault-tree style reliability model with latent shared subsystems.

    Parameters
    ----------
    n_gpus : int
        Total number of GPUs.
    subsystem_coverage : dict[int, set[int]]
        Maps subsystem id -> set of GPU indices that die if this subsystem fails.
    p_fail : np.ndarray, shape (m,)
        Prior failure probability for each subsystem (small, e.g. 1e-2).
    connectivity_edges : list[tuple[int, int]]
        Undirected edges of the physical GPU interconnect graph.
    job_gpus : set[int]
        GPUs that the job requires.
    min_healthy : int
        Job fails if fewer than min_healthy requested GPUs are healthy.
    require_connected : bool
        If True, also require the healthy requested GPUs to form a connected subgraph.
    """
    def __init__(self, n_gpus, subsystem_coverage, p_fail, connectivity_edges,
                 job_gpus, min_healthy, require_connected=True):
        self.n_gpus = n_gpus
        self.m = len(subsystem_coverage)
        self.coverage = subsystem_coverage
        self.p_fail = np.asarray(p_fail, dtype=float)
        self.p_work = 1.0 - self.p_fail
        self.edges = connectivity_edges
        self.job_gpus = set(job_gpus)
        self.min_healthy = min_healthy
        self.require_connected = require_connected

        # Precompute adjacency list for connectivity checks
        self.adj = {i: set() for i in range(n_gpus)}
        for u, v in connectivity_edges:
            self.adj[u].add(v)
            self.adj[v].add(u)

    def gpu_health(self, z):
        """
        Compute GPU health vector from subsystem state z.

        Parameters
        ----------
        z : np.ndarray, shape (m,) or (batch, m)
            1 = subsystem working, 0 = failed.

        Returns
        -------
        healthy : np.ndarray, shape (n_gpus,) or (batch, n_gpus)
            1 = GPU healthy, 0 = dead.
        """
        z = np.atleast_2d(z)
        batch = z.shape[0]
        healthy = np.ones((batch, self.n_gpus), dtype=int)
        for sid, gpus in self.coverage.items():
            failed = (z[:, sid] == 0)
            for g in gpus:
                healthy[failed, g] = 0
        return healthy.squeeze()

    def is_connected(self, healthy_gpus):
        """
        Check if the healthy GPUs in the job set form a connected subgraph.

        Parameters
        ----------
        healthy_gpus : np.ndarray, shape (n_gpus,)
            Binary health vector.
        """
        alive_in_job = self.job_gpus & set(np.where(healthy_gpus)[0])
        if len(alive_in_job) == 0:
            return False
        start = next(iter(alive_in_job))
        visited = set([start])
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v in alive_in_job and v not in visited:
                    visited.add(v)
                    queue.append(v)
        return len(visited) == len(alive_in_job)

    def phi(self, z):
        """
        System function: 1 = job succeeds, 0 = job fails.

        Parameters
        ----------
        z : np.ndarray, shape (m,)
            Subsystem state.
        """
        healthy = self.gpu_health(z)
        job_alive_count = int(np.sum(healthy[list(self.job_gpus)]))
        if job_alive_count < self.min_healthy:
            return 0
        if self.require_connected and not self.is_connected(healthy):
            return 0
        return 1

    def exact_pfail(self):
        """
        Exact failure probability by enumerating all 2^m subsystem states.

        Returns
        -------
        p_fail_exact : float
        p_star : dict
            Mapping state_tuple -> probability (for constructing exact target).
        states : list
            List of all failure states.
        """
        total = 0.0
        p_star = {}
        states = []
        for bits in product([0, 1], repeat=self.m):
            z = np.array(bits, dtype=int)
            p = float(np.prod(np.where(z, self.p_work, self.p_fail)))
            if self.phi(z) == 0:
                total += p
                p_star[bits] = p
                states.append(z)
        return total, p_star, states

    def sample_prior(self, n_samples, rng=None):
        """Sample n_samples from the prior P(Z)."""
        if rng is None:
            rng = np.random.default_rng()
        return (rng.random((n_samples, self.m)) > self.p_fail).astype(int)

    def sample_tilted(self, n_samples, p_tilt, rng=None):
        """Sample n_samples from independent Bernoulli(p_tilt) proposal."""
        if rng is None:
            rng = np.random.default_rng()
        return (rng.random((n_samples, self.m)) > p_tilt).astype(int)

    def likelihood_ratio(self, z, p_tilt):
        """
        Likelihood ratio P(Z=z) / Q(Z=z) for independent tilted proposal Q.

        Parameters
        ----------
        z : np.ndarray, shape (batch, m)
        p_tilt : np.ndarray, shape (m,)
        """
        z = np.atleast_2d(z)
        q_fail = p_tilt
        q_work = 1.0 - p_tilt
        lr = np.prod(np.where(z, self.p_work / q_work, self.p_fail / q_fail), axis=1)
        return lr
