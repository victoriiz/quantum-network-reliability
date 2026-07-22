"""
Three experiments on the dynamic model

EXP A: establish exact trajectory-space ceiling for classical tilt family and verify
EXP B: amplitude estimation on path space, does quadratic advantage survive the cost of reversible simulation?
EXP C: amplify then prepare -> build p* over trajectories via Grover rotaion, verify it is p*, use for conditional stats
"""

import numpy as np
import sys, os
from model import DynamicConfig, DynamicModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from estimators import exact_traj_ceiling

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- instance
N = 3
m = DynamicModel(DynamicConfig(
    N=N, T=6, c=np.array([2.0, 1.0, 1.0]), c_min=2.0,
    a0=np.full(N, 0.0012), gamma=np.full(N, 0.3),
    b0=np.full(N, 0.3), eta=np.full(N, 0.5), name="dyn-3-het-rare"))
v = m.validate()
print(f"[{v['name']}] p_T={m.p_fail:.4e}  F={v['n_fail_states']}/{v['n_states']}"
      f"  rows_ok={v['rows_sum_to_1']}  dp=sim={v['dp_matches_sim']}")
assert v["rows_sum_to_1"] and v["dp_matches_sim"]
P, in_F, start, T, nS, pT = (m.Pmat, m.in_F, m.start_idx, m.T,
                             m.n_states, m.p_fail)
 
 
# ============================================================== EXP A
# Support-safe per-component odds-tilt family: scales every failure rate a_i(x) -> min(1, lam*a_i(x))
def odds_tilt_kernel(model, lam):
    Q = np.zeros_like(P); S = model.states
    for i in range(nS):
        x = S[i]
        a0v = model._a_i(x)
        a = lam * a0v / (1.0 - a0v + lam * a0v)          # odds-tilt
        b = model._b_i(x)
        for j in range(nS):
            y = S[j]; p = 1.0
            for k in range(N):
                if   x[k] == 1 and y[k] == 1: p *= (1 - a[k])
                elif x[k] == 1 and y[k] == 0: p *= a[k]
                elif x[k] == 0 and y[k] == 1: p *= b[k]
                else:                         p *= (1 - b[k])
            Q[i, j] = p
    return Q
 
lams = np.geomspace(1.0, 400.0, 200)
ceil = exact_traj_ceiling(m, odds_tilt_kernel, lams)
print(f"\nEXP A  exact trajectory-space ceiling (odds-tilt family): "
      f"VRF_max = {ceil['vrf']:.1f}x at lambda = {ceil['params']:.1f}")
print(f"       (a scalar odds knob cannot express WHICH component to fail;"
      f" per-component lambda_i is the natural next family, priced the same way)")
 
vrfs = []
for l in lams:
    Q = odds_tilt_kernel(m, l)
    R = np.where(Q > 0, P * P / Q, 0.0)
    M = np.zeros(nS)
    for _ in range(T):
        M = R[:, in_F == 1].sum(axis=1) + R[:, in_F == 0] @ M[in_F == 0]
        M[in_F == 1] = 0.0
    var = M[start] - pT ** 2
    vrfs.append(pT * (1 - pT) / var if var > 0 else np.nan)
plt.figure(figsize=(6.5, 4))
plt.semilogx(lams, vrfs, lw=2)
plt.scatter([ceil["params"]], [ceil["vrf"]], color="crimson", zorder=5,
            label=f"ceiling {ceil['vrf']:.1f}x @ \u03bb={ceil['params']:.0f}")
plt.xlabel("failure-odds boost \u03bb"); plt.ylabel("exact VRF")
plt.title("EXP A: exact trajectory-space ceiling (no sampling)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/expA_traj_ceiling.png", dpi=140); plt.close()
 
# ============================================================== EXP B
theta = np.arcsin(np.sqrt(pT))
def qae_rmse(J, shots=60, reps=60, seed=0):
    rng = np.random.default_rng(seed)
    ms = [0] + [2 ** j for j in range(J)]
    grid = np.linspace(theta * 0.2, min(np.pi / 2, theta * 5), 4000)
    ests = []
    for _ in range(reps):
        counts = [(rng.binomial(shots, np.sin((2 * mm + 1) * theta) ** 2), mm)
                  for mm in ms]
        ll = np.zeros_like(grid)
        for h, mm in counts:
            p1 = np.sin((2 * mm + 1) * grid) ** 2
            ll += h * np.log(p1 + 1e-12) + (shots - h) * np.log(1 - p1 + 1e-12)
        ests.append(np.sin(grid[np.argmax(ll)]) ** 2)
    ests = np.array(ests)
    return sum(shots * (2 * mm + 1) for mm in ms), \
        np.sqrt(np.mean((ests - pT) ** 2)) / pT
 
qx, qy = np.array([qae_rmse(J) for J in range(2, 9)]).T
cx = np.logspace(3.2, 6.8, 8); cy = np.sqrt((1 - pT) / (cx * pT))
sq = np.polyfit(np.log(qx), np.log(qy), 1)[0]
print(f"\nEXP B  path-space QAE slope {sq:.2f} (theory -1) vs classical -0.50;"
      f" at RMSE~{qy[-1]*100:.2f}%: {int(qx[-1]):,} oracle calls vs "
      f"~{int((1-pT)/(pT*qy[-1]**2)):,} sample paths")
plt.figure(figsize=(6.5, 4))
plt.loglog(cx, cy, "k--", lw=2, label="classical traj-MC (slope -0.50)")
plt.loglog(qx, qy, "o-", lw=2, color="seagreen",
           label=f"path-space QAE (slope {sq:.2f})")
plt.xlabel("budget: oracle calls / sample paths")
plt.ylabel("relative RMSE of p_T")
plt.title("EXP B: quadratic query separation on trajectory space")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(f"{OUT}/expB_pathspace_qae.png", dpi=140); plt.close()
 
# ============================================================== EXP C
paths, probs, hits, ht = [], [], [], []
def enum(x, t, logp, hist):
    if in_F[x]:
        probs.append(np.exp(logp)); hits.append(1); ht.append(t); return
    if t == T:
        probs.append(np.exp(logp)); hits.append(0); ht.append(np.nan); return
    for y in range(nS):
        if P[x, y] > 0:
            enum(y, t + 1, logp + np.log(P[x, y]), hist + [y])
enum(start, 0, 0.0, [start])
probs = np.array(probs); hits = np.array(hits); ht = np.array(ht)
assert abs(probs.sum() - 1) < 1e-9 and abs(probs[hits == 1].sum() - pT) < 1e-9
pstar = probs * hits / pT
ks = np.arange(0, int(np.pi / (4 * theta)) + 8)
succ = np.sin((2 * ks + 1) * theta) ** 2
tvs = []
for k in ks:
    s2, c2 = np.sin((2 * k + 1) * theta) ** 2, np.cos((2 * k + 1) * theta) ** 2
    meas = s2 * probs * hits / pT + c2 * probs * (1 - hits) / (1 - pT)
    tvs.append(0.5 * np.abs(meas - pstar).sum())
k_star = int(np.round(np.pi / (4 * theta) - 0.5))
exact_cond = float(np.nansum(ht * probs * hits) / pT)
rng = np.random.default_rng(0)
est_cond = float(np.nanmean(ht[rng.choice(len(probs), 500, p=pstar)]))
print(f"\nEXP C  amplify-and-prepare: k*={k_star} rounds (~1/\u221ap={1/np.sqrt(pT):.0f});"
      f" success {succ[k_star]:.4f}; TV(measured,p*)={tvs[k_star]:.2e}")
print(f"       E[first-hit time|fail]: exact {exact_cond:.3f}, "
      f"500 amplified samples {est_cond:.3f}")
print(f"       cost: 500x{k_star}={500*k_star:,} coherent sims vs "
      f"classical rejection ~{int(500/pT):,} paths")
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(ks, succ, "o-", color="seagreen")
ax[0].axvline(k_star, color="crimson", ls="--", label=f"k*={k_star}")
ax[0].set_xlabel("Grover rounds k"); ax[0].set_ylabel("P(measure failing path)")
ax[0].set_title("amplification of the failure subspace"); ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].semilogy(ks, tvs, "o-", color="navy")
ax[1].axvline(k_star, color="crimson", ls="--")
ax[1].set_xlabel("Grover rounds k"); ax[1].set_ylabel("TV(measured, p*)")
ax[1].set_title("prepared state \u2192 the ideal trajectory sampler")
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/expC_amplify_prepare.png", dpi=140)
plt.close()
print(f"\nplots -> {OUT}/")