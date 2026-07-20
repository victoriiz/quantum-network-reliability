"""
pipeline/model.py -- LAYER 1 only: the model.

This is the ONLY file you rewrite for a new static problem. It exposes a
minimal contract the rest of the pipeline depends on:

    .n_states           number of enumerable states
    .p                  np.ndarray (n_states,)  prior p(x)
    .failmask           np.ndarray (n_states,)  1_fail(x) in {0,1}
    .p_fail             float                   exact P_fail
    .n_components       int                     for per-component proposals
    .component_bits     np.ndarray (n_states, n_components)  x as bits

Everything downstream (ideal, proposals, audit, ceiling, report) is
written against THIS contract and never needs to change when the model does.
"""
from dataclasses import dataclass, field
from itertools import product
import numpy as np


@dataclass
class FaultTreeConfig:
    """An instance is DATA, not code. Change these to change the problem."""
    m: int                              # number of subsystems (components)
    p_fail: np.ndarray                  # per-subsystem failure prob (len m)
    coverage: dict                      # subsystem -> set of GPUs it kills
    n_gpu: int
    job: set                            # GPUs the job requests
    min_healthy: int                    # fail if fewer than this survive
    name: str = "unnamed"


class FaultTreeModel:
    def __init__(self, cfg: FaultTreeConfig):
        self.cfg = cfg
        self.n_components = cfg.m
        # --- enumerate + VECTORIZE ONCE (demo recomputed per call) --------
        bits = np.array(list(product([0, 1], repeat=cfg.m)), dtype=np.int8)
        self.component_bits = bits                     # z[j]=1 means WORKS
        self.n_states = len(bits)
        # p(z): product over subsystems, computed as a matrix op
        pw = np.where(bits == 1, 1 - cfg.p_fail, cfg.p_fail)
        self.p = pw.prod(axis=1)
        # 1_fail(z): vectorized health + threshold
        self.failmask = self._failmask(bits)
        self.p_fail = float(self.p @ self.failmask)

    def _failmask(self, bits):
        """1 if the job fails in state z. Vectorized over all states."""
        # gpu_health[s, i] = 1 iff every subsystem covering GPU i works in state s
        health = np.ones((len(bits), self.cfg.n_gpu), dtype=np.int8)
        for j, covered in self.cfg.coverage.items():
            for i in covered:
                health[:, i] &= bits[:, j]             # AND in this subsystem
        job_idx = sorted(self.cfg.job)
        n_healthy = health[:, job_idx].sum(axis=1)
        return (n_healthy < self.cfg.min_healthy).astype(np.int8)

    def validate(self):
        assert abs(self.p.sum() - 1.0) < 1e-9, "prior doesn't sum to 1"
        n_fail = int(self.failmask.sum())
        rare = self.p_fail < 0.01
        concentrated = n_fail < self.n_states * 0.5
        return {
            "name": self.cfg.name,
            "p_fail": self.p_fail,
            "n_fail_states": n_fail,
            "n_states": self.n_states,
            "is_rare": rare,
            "is_concentrated": concentrated,
            "usable": rare and concentrated,
        }