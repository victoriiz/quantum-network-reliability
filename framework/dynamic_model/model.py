"""
model is now a Markov chain over T steps, solved
by backward DP instead of enumeration. 

Contract the runner expects (mirror of the static one):
    .n_states, .T
    .Pmat            (n_states, n_states) transition matrix
    .in_F            (n_states,) 1 if state is in failure set
    .start_idx       index of the all-working state
    .p_fail          exact p_T from DP
    .H               list of value functions h_0..h_T   (for the h-transform)
"""
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass
class DynamicConfig:
    N: int                    # components
    T: int                    # mission horizon
    c: np.ndarray             # capacities (HETEROGENEOUS = the hardness dial)
    c_min: float              # fail if total capacity < c_min
    a0: np.ndarray            # nominal per-step failure prob
    gamma: np.ndarray         # overload sensitivity
    b0: np.ndarray            # nominal repair prob
    eta: np.ndarray           # repair congestion
    name: str = "unnamed"


class DynamicModel:
    def __init__(self, cfg: DynamicConfig):
        self.cfg = cfg
        self.T = cfg.T
        self.states = np.array(list(product([0, 1], repeat=cfg.N)),
                               dtype=np.int8)
        self.n_states = len(self.states)
        self.cap = self.states @ cfg.c
        self.c_nom = cfg.c.sum()

        # ---- LAYER 1a: the failure set F ---------------------------------
        self.in_F = (self.cap < cfg.c_min).astype(int)  

        # ---- LAYER 1b: the transition matrix -----------------------------
        self.Pmat = self._build_transition_matrix()

        # ---- LAYER 1c: backward DP for p_fail and H ----------------------
        self.H = self._backward_dp()
        self.start_idx = int(np.where((self.states == 1).all(axis=1))[0][0])
        self.p_fail = float(self.H[self.T][self.start_idx])

    def _a_i(self, x):
        """Per-component failure prob given state x (overload cascade)."""
        overload = np.maximum(self.c_nom / max(x @ self.cfg.c, 1e-9) - 1.0, 0)
        return np.minimum(1.0, self.cfg.a0 + self.cfg.gamma * overload)

    def _b_i(self, x):
        """Per-component repair prob given state x (repair congestion)."""
        return self.cfg.b0 / (1.0 + self.cfg.eta * (self.cfg.N - x.sum()))

    def _trans_prob(self, x, y):
        """P(x -> y): product of per-component transitions."""
        a, b = self._a_i(x), self._b_i(x)
        p = 1.0
        for i in range(self.cfg.N):
            if x[i] == 1 and y[i] == 1: p *= (1-a[i])
            elif x[i] == 1 and y[i] == 0: p *= a[i]
            elif x[i] == 0 and y[i] == 1: p *= b[i]
            else: p *= (1 - b[i])
        return p

    def _build_transition_matrix(self):
        S = self.states
        return np.array([[self._trans_prob(S[i], S[j])
                          for j in range(self.n_states)]
                         for i in range(self.n_states)])

    def _backward_dp(self):
        """h_t(x) = Pr(fail by mission end | in x, t steps elapsed).
        h_0 = 1[x in F]; h_{t+1} = P @ h_t, with F absorbing."""
        h = self.in_F.astype(float).copy()
        H = [h.copy()]
        for _ in range(self.T):
            h_next = self.Pmat @ h
            h_next[self.in_F == 1] = 1.0
            h = h_next
            H.append(h.copy())

        return H # H[t] = h_t, len = T+1

    # ---- LAYER 2: the h-transform (ideal sequential proposal) -----------
    def tilted_transition(self, x_idx, steps_left):
        """q*(x -> .) proportional to P(x,.) * h_{steps_left-1}(.)."""
        if steps_left == 0:
            return None
        w = self.Pmat[x_idx] * self.H[steps_left-1]
        return w / w.sum() if w.sum() > 0 else self.Pmat[x_idx]
    
    def naive_sim(self, n_paths=200_000, seed=0):
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_paths):
            x = self.start_idx
            failed = int(self.in_F[x])
            for _ in range(self.T):
                x = rng.choice(self.n_states, p=self.Pmat[x])
                if self.in_F[x]:
                    failed = 1
            hits += failed
        return hits / n_paths

    def validate(self):
        rows_ok = np.allclose(self.Pmat.sum(axis=1), 1.0)   # move-3 test
        sim = self.naive_sim(50_000)                        # move-4 test
        dp_sim_agree = abs(sim - self.p_fail) < 3 * (self.p_fail *
                        (1 - self.p_fail) / 50_000) ** 0.5 + 1e-6
        return {"name": self.cfg.name, "p_fail": self.p_fail,
                "n_fail_states": int(self.in_F.sum()),
                "n_states": self.n_states,
                "is_rare": self.p_fail < 0.01,
                "rows_sum_to_1": bool(rows_ok),
                "dp_matches_sim": bool(dp_sim_agree), "sim": sim}
    
if __name__ == "__main__":
    cfg = DynamicConfig(
        N=3, T=5, c=np.array([1.0, 1.0, 1.0]), c_min=1.5,
        a0=np.array([0.05] * 3), gamma=np.array([0.3] * 3),
        b0=np.array([0.2] * 3), eta=np.array([0.5] * 3), name="test-3")
    m = DynamicModel(cfg)
    v = m.validate()
    for k, val in v.items():
        print(f"  {k}: {val}")
