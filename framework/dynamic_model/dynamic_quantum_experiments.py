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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ----- instance ------
N = 3
cfg = DynamicConfig(
    N=N, T=6,
    c=np.array([2.0, 1.0, 1.0]), c_min=2.0,    
    a0=np.full(N, 0.0012), gamma=np.full(N, 0.3),
    b0=np.full(N, 0.3), eta=np.full(N, 0.5),
    name="dyn-3-het-rare")
m = DynamicModel(cfg)
v = m.validate()
print(f"[{v['name']}] p_T = {m.p_fail:.4e}  F={v['n_fail_states']}/{v['n_states']}"
      f"  rows_ok={v['rows_sum_to_1']}  dp=sim={v['dp_matches_sim']}")
assert v["rows_sum_to_1"] and v["dp_matches_sim"]
P, in_F, start, T = m.Pmat, m.in_F, m.start_idx, m.T
nS = m.n_states
pT = m.p_fail

# ------ EXP A --------
# "failure-rate boost family": scales every failure rate a_i(x) -> min(1, lam*a_i(x))

def boosted_Pmat(lam):
    Q = np.zeros_like(P)
    S = m.states
    for i in range(nS):
        x = S[i]
        a0v = m._a_i(x)
        a = lam * a0v / (1.0 - a0v + lam*a0v)
        b = m._b_i(x)
        for j in range(nS):
            y = S[j]; p = 1.0
            for k in range(N):
                if x[k] == 1 and y[k] == 1: p *= (1-a[k])
                elif x[k] == 1 and y[k] == 0: p *= a[k]
                elif x[k] == 0 and y[k] == 1: p *= b[k]
                else: p *= (1-b[k])
            Q[i, j] = p
    return Q
        
def exact_var_traj(Q):
    R = np.where(Q > 0, P * P / Q, 0.0)          # ratio matrix P^2/Q
    Mprev = np.zeros(nS)
    for _ in range(T):
        Mnew = R[:, in_F == 1].sum(axis=1) + R[:, in_F == 0] @ Mprev[in_F == 0]
        Mnew[in_F == 1] = 0.0                    # never stepped from inside F
        Mprev = Mnew
    return Mprev[start] - pT ** 2
 
lams = np.geomspace(1.0, 400.0, 240)
vrfs = np.array([pT * (1 - pT) / exact_var_traj(boosted_Pmat(l)) for l in lams])
best = np.argmax(vrfs)
print(f"\nEXP A  exact trajectory-space ceiling (rate-boost family):"
      f"  VRF_max = {vrfs[best]:.1f}x at lambda = {lams[best]:.1f}"
      f"   (naive lambda=1: {vrfs[0]:.2f}x)")
 
plt.figure(figsize=(6.5, 4))
plt.semilogy(lams, vrfs, lw=2)
plt.scatter([lams[best]], [vrfs[best]], color="crimson", zorder=5,
            label=f"ceiling {vrfs[best]:.0f}x @ \u03bb={lams[best]:.1f}")
plt.xlabel("failure-rate boost \u03bb"); plt.ylabel("exact VRF (log)")
plt.title("EXP A: exact trajectory-space ceiling (no sampling)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/failure_boost_ceiling.png", dpi=140); plt.close()
 
# sanity: audit the best boosted proposal by trajectory IS (stop at entry)
def traj_audit(Q, n_paths=60_000, n_trials=8, seed=0):
    ests = np.empty(n_trials)
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t); ws = np.empty(n_paths)
        for k in range(n_paths):
            x = start; logw = 0.0; hit = 0
            for _ in range(T):
                y = rng.choice(nS, p=Q[x])
                logw += np.log(P[x, y] + 1e-300) - np.log(Q[x, y] + 1e-300)
                x = y
                if in_F[x]: hit = 1; break
            ws[k] = np.exp(logw) * hit
        ests[t] = ws.mean()
    var = ests.var(ddof=1) * n_paths
    return ests.mean(), pT * (1 - pT) / var
 
mn, vr = traj_audit(boosted_Pmat(lams[best]))
print(f"       audit of best boost: mean={mn:.3e} (exact {pT:.3e}), "
      f"measured VRF={vr:.1f}x (exact {vrfs[best]:.1f}x)")

# ------ EXP B --------
# AE on path space: oracle = one coherent T-step simulation; marked
# amplitude sqrt(pT). MLE-QAE on exact Born probabilities vs classical
# trajectory MC. Slopes + query counts.
theta = np.arcsin(np.sqrt(pT))
def qae_rmse(budget_powers, shots=60, reps=60, seed=0):
    rng = np.random.default_rng(seed)
    ms = [0] + [2 ** j for j in range(budget_powers)]
    grid = np.linspace(theta * 0.2, min(np.pi / 2, theta * 5), 4000)
    ests = []
    for r in range(reps):
        counts = [(rng.binomial(shots, np.sin((2 * mm + 1) * theta) ** 2), mm)
                  for mm in ms]
        ll = np.zeros_like(grid)
        for h, mm in counts:
            p1 = np.sin((2 * mm + 1) * grid) ** 2
            ll += h * np.log(p1 + 1e-12) + (shots - h) * np.log(1 - p1 + 1e-12)
        ests.append(np.sin(grid[np.argmax(ll)]) ** 2)
    ests = np.array(ests)
    queries = sum(shots * (2 * mm + 1) for mm in ms)
    return queries, np.sqrt(np.mean((ests - pT) ** 2)) / pT
 
qx, qy = [], []
for J in range(2, 9):
    q, e = qae_rmse(J)
    qx.append(q); qy.append(e)
qx, qy = np.array(qx), np.array(qy)
cx = np.logspace(3.2, 6.8, 8)
cy = np.sqrt((1 - pT) / (cx * pT))          # classical trajectory-MC RMSE
sq = np.polyfit(np.log(qx), np.log(qy), 1)[0]
sc = -0.5
print(f"\nEXP B  path-space QAE: slope {sq:.2f} (theory -1) vs classical "
      f"{sc:.2f}; at RMSE~{qy[-1]*100:.2f}%: QAE {qx[-1]:,} queries vs "
      f"classical ~{(1-pT)/(pT*(qy[-1])**2):,.0f} paths")
plt.figure(figsize=(6.5, 4))
plt.loglog(cx, cy, "k--", lw=2, label=f"classical traj-MC (slope {sc:.2f})")
plt.loglog(qx, qy, "o-", lw=2, color="seagreen",
           label=f"QAE on path space (slope {sq:.2f})")
plt.xlabel("budget: oracle calls (coherent T-step sims) / sample paths")
plt.ylabel("relative RMSE of p_T")
plt.title("EXP B: quadratic query separation on trajectory space")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(f"{OUT}/expB_pathspace_qae.png", dpi=140); plt.close()
 
# ------ EXP C --------
# Amplify-and-prepare: enumerate stopped trajectories, amplitudes
# sqrt(P(tau)); Grover-rotate toward the failing subspace; verify the
# measurement distribution equals p*(tau); use it for a conditional
# statistic classical rejection would pay 1/pT for.
paths, probs, hits, hit_time = [], [], [], []
def enum(x, t, logp, hist):
    if in_F[x]:
        paths.append(tuple(hist)); probs.append(np.exp(logp))
        hits.append(1); hit_time.append(t); return
    if t == T:
        paths.append(tuple(hist)); probs.append(np.exp(logp))
        hits.append(0); hit_time.append(np.nan); return
    for y in range(nS):
        if P[x, y] > 0:
            enum(y, t + 1, logp + np.log(P[x, y]), hist + [y])
enum(start, 0, 0.0, [start])
probs = np.array(probs); hits = np.array(hits); ht = np.array(hit_time)
assert abs(probs.sum() - 1) < 1e-9 and abs(probs[hits == 1].sum() - pT) < 1e-9
pstar = probs * hits / pT                       # ideal over trajectories
 
ks = np.arange(0, int(np.pi / (4 * theta)) + 8)
succ = np.sin((2 * ks + 1) * theta) ** 2
tvs = []
for k in ks:
    s2, c2 = np.sin((2 * k + 1) * theta) ** 2, np.cos((2 * k + 1) * theta) ** 2
    meas = s2 * probs * hits / pT + c2 * probs * (1 - hits) / (1 - pT)
    tvs.append(0.5 * np.abs(meas - pstar).sum())
k_star = int(np.round(np.pi / (4 * theta) - 0.5))
print(f"\nEXP C  amplify-and-prepare: k* = {k_star} Grover rounds "
      f"(~O(1/\u221ap) = {1/np.sqrt(pT):.0f});  success prob at k*: "
      f"{succ[k_star]:.4f};  TV(measured, p*) at k*: {tvs[k_star]:.2e}")
# conditional statistic: E[first-hit time | failure]
exact_cond = float(np.nansum(ht * probs * hits) / pT)
rng = np.random.default_rng(0)
draw = rng.choice(len(probs), size=500, p=pstar)   # 500 amplified samples
est_cond = float(np.nanmean(ht[draw]))
print(f"       E[first-hit time | fail]: exact {exact_cond:.3f}, "
      f"from 500 amplified samples {est_cond:.3f}")
print(f"       classical rejection cost for 500 conditioned samples: "
      f"~{500/pT:,.0f} paths;  amplified cost: 500 x {k_star} rounds "
      f"= {500*k_star:,} coherent sims")
 
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(ks, succ, "o-", color="seagreen")
ax[0].axvline(k_star, color="crimson", ls="--", label=f"k* = {k_star}")
ax[0].set_xlabel("Grover rounds k"); ax[0].set_ylabel("P(measure failing path)")
ax[0].set_title("amplification of the failure subspace"); ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].semilogy(ks, tvs, "o-", color="navy")
ax[1].axvline(k_star, color="crimson", ls="--")
ax[1].set_xlabel("Grover rounds k")
ax[1].set_ylabel("TV( measured , p* )  (log)")
ax[1].set_title("prepared state \u2192 the ideal trajectory sampler")
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/expC_amplify_prepare.png", dpi=140)
plt.close()
print(f"\nplots -> {OUT}/")
