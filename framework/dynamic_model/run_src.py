"""
SRC experiment driver: the expressiveness ladder.
Goes in: experiments/src26/run_ceiling_ladder.py   (or framework/dynamic_model/run_src.py)

  0. correctness      recursion vs brute force; DP vs naive simulation
  1. instrument gate  h-transform audits to ~zero bias, ESS ~ 1.0
  2. scalar ceiling   1 parameter
  3. per-component    N parameters      <- the new result
  4. per-comp+repair  2N parameters
  5. payoff           p_T -> r_f -> A -> N_prod

Run:  python run_ceiling_ladder.py --N 8 --job-hours 2 --out results.json
"""

import argparse
import json
import time

import numpy as np

from model import DynamicConfig, DynamicModel
from proposals import build_kernel, h_transform_sequence, kernel_from_theta
from ceiling import (exact_traj_variance, exact_traj_vrf, family_ceiling,
                     brute_force_variance, vrf_from_var, run_ladder)
from audit import audit_trajectory
from delta_calibration import delta_instance, capacity_plan


def build_model(spec):
    cfg = DynamicConfig(
        N=spec["N"], T=spec["T"], c=spec["c"], c_min=spec["c_min"],
        a0=spec["a0"], gamma=spec["gamma"], b0=spec["b0"], eta=spec["eta"],
        name=spec["name"])
    return DynamicModel(cfg)


def check_recursion(seed=0):
    """Gate 0a: the second-moment recursion vs brute-force path enumeration.

    Tiny instance only -- brute force is exponential in T. If this disagrees,
    every ceiling in the paper is wrong, so it runs first and it runs loud.
    """
    cfg = DynamicConfig(
        N=3, T=3, c=np.array([2.0, 1.0, 1.0]), c_min=2.0,
        a0=np.full(3, 0.05), gamma=np.full(3, 0.3),
        b0=np.full(3, 0.3), eta=np.full(3, 0.5), name="recursion-check")
    m = DynamicModel(cfg)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(5):
        lam = np.exp(rng.normal(0, 0.8, size=3))
        Q = build_kernel(m, lam)
        v_rec = exact_traj_variance(m, Q)
        v_bf = brute_force_variance(m, Q)
        worst = max(worst, abs(v_rec - v_bf) / max(abs(v_bf), 1e-300))
    return {"max_rel_err": worst, "passed": worst < 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--node-type", default="A100")
    ap.add_argument("--step-hours", type=float, default=0.25)
    ap.add_argument("--job-hours", type=float, default=2.0)
    ap.add_argument("--request-frac", type=float, default=0.25)
    ap.add_argument("--big-nodes", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.30)
    ap.add_argument("--eta", type=float, default=0.50)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sla-gpus", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="src_results.json")
    ap.add_argument("--skip-recursion-check", action="store_true")
    args = ap.parse_args()

    out = {"args": vars(args)}

    # ---- 0a. recursion correctness --------------------------------------
    if not args.skip_recursion_check:
        print("[0a] recursion vs brute force ...")
        rc = check_recursion(args.seed)
        out["recursion_check"] = rc
        print(f"     max rel err {rc['max_rel_err']:.3e}  "
              f"{'PASS' if rc['passed'] else 'FAIL'}")
        if not rc["passed"]:
            raise SystemExit("recursion check FAILED -- stop, do not report ceilings")

    # ---- 0b. build and validate the Delta-calibrated instance ------------
    print(f"[0b] building Delta-calibrated instance, N={args.N} ...")
    spec = delta_instance(N=args.N, node_type=args.node_type,
                          step_hours=args.step_hours, job_hours=args.job_hours,
                          big_nodes=args.big_nodes,
                          request_frac=args.request_frac,
                          gamma=args.gamma, eta=args.eta)
    t0 = time.time()
    model = build_model(spec)
    build_s = time.time() - t0
    val = model.validate()
    out["instance"] = {
        "name": spec["name"], "N": spec["N"], "T": spec["T"],
        "c": spec["c"].tolist(), "c_min": spec["c_min"],
        "c_nom": float(spec["_c_nom"]),
        "a0": float(spec["a0"][0]), "b0": float(spec["b0"][0]),
        "gamma": args.gamma, "eta": args.eta,
        "step_hours": args.step_hours, "job_hours": args.job_hours,
        "build_seconds": build_s,
    }
    out["validation"] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else v) for k, v in val.items()}
    print(f"     p_fail = {model.p_fail:.6e}   states {model.n_states}   "
          f"fail states {int(model.in_F.sum())}   build {build_s:.1f}s")
    print(f"     rare (<1e-2): {val['is_rare']}   "
          f"rows sum to 1: {val['rows_sum_to_1']}   "
          f"DP matches sim: {val['dp_matches_sim']}")
    if not val["is_rare"]:
        print("     WARNING: not a rare instance. Raise --request-frac.")

    # ---- 1. instrument gate: the h-transform -----------------------------
    print("[1] h-transform audit (expect bias ~0, ESS ~1.0) ...")
    Q_seq = h_transform_sequence(model)
    a_ideal = audit_trajectory(model, Q_seq=Q_seq, n_paths=2_000,
                               n_trials=args.trials, seed=args.seed)
    out["h_transform_audit"] = a_ideal
    print(f"     bias {a_ideal['bias']*100:+.4f}%   ESS {a_ideal['ess']:.4f}")

    # ---- 1b. naive baseline ---------------------------------------------
    print("[1b] naive Monte Carlo baseline ...")
    a_naive = audit_trajectory(model, Q=model.Pmat.copy(),
                               n_paths=args.paths, n_trials=args.trials,
                               seed=args.seed)
    out["naive_audit"] = a_naive
    print(f"     bias {a_naive['bias']*100:+.3f}%   VRF {a_naive['vrf']:.3f}")

    # ---- 2-4. the ladder -------------------------------------------------
    print("[ladder] optimising nested families (warm-started) ...")
    t0 = time.time()
    lad = run_ladder(model, n_restarts=args.restarts, seed=args.seed,
                     maxiter=args.maxiter, verbose=True)
    print(f"     monotone: {lad['monotone']}   "
          f"all symmetric: {lad['all_symmetric']}   "
          f"({time.time() - t0:.1f}s)")

    ladder = {}
    for family, res in lad["families"].items():
        Q = kernel_from_theta(model, family, res["theta"])
        aud = audit_trajectory(model, Q=Q, n_paths=args.paths,
                               n_trials=args.trials, seed=args.seed)
        ladder[family] = {
            "n_params": res["n_params"],
            "ceiling_vrf": res["vrf"],
            "ceiling_var": res["var"],
            "sym_violation": res["sym_violation"],
            "sym_ok": bool(res["sym_ok"]),
            "lam_fail": np.asarray(res["lam_fail"]).tolist(),
            "lam_repair": np.asarray(res["lam_repair"]).tolist(),
            "audit": aud,
        }
        print(f"     {family:<17} ceiling {res['vrf']:>9.4f}   "
              f"sampled {aud['vrf']:>8.4f}   bias {aud['bias']*100:+.3f}%   "
              f"ESS {aud['ess']:.4f}")
    out["ladder"] = ladder
    out["ladder_invariants"] = {"monotone": lad["monotone"],
                                "all_symmetric": lad["all_symmetric"]}
    if not lad["monotone"]:
        print("     !! ceilings are NOT reportable: nested family regressed")

    # ---- 5. payoff -------------------------------------------------------
    print("[5] capacity-planning payoff ...")
    g = int(spec["c"][0])
    plan = capacity_plan(p_T=model.p_fail, T=model.T,
                         step_hours=args.step_hours,
                         mttr_hours=spec["_rates"]["mttr_hours"],
                         n_target=args.sla_gpus, g=g)
    out["capacity_plan"] = plan.__dict__
    print(f"     p_T {plan.p_T:.4e}  ->  r_f {plan.r_f:.4e}/h  ->  "
          f"A {plan.A:.5f}  ->  N_prod {plan.n_prod} for {plan.n_target} "
          f"({plan.overprovision*100:.2f}% overprovision)")

    # ---- summary table ---------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{'family':<20}{'params':>8}{'exact ceiling':>16}{'sampled VRF':>16}")
    print("-" * 66)
    print(f"{'naive MC':<20}{0:>8}{1.0:>16.3f}{a_naive['vrf']:>16.3f}")
    for fam, r in ladder.items():
        print(f"{fam:<20}{r['n_params']:>8}{r['ceiling_vrf']:>16.3f}"
              f"{r['audit']['vrf']:>16.3f}")
    print(f"{'h-transform':<20}{'--':>8}{'inf':>16}{'inf':>16}")
    print("=" * 66)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()