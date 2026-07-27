"""
Delta / DeltaAI calibration and the capacity-planning payoff.
Goes in: framework/dynamic_model/delta_calibration.py

SOURCE OF EVERY NUMBER BELOW:
  Shengkun Cui, Archit Patke, Hung Nguyen, Aditya Ranjan, Ziheng Chen,
  Phuong Cao, Gregory Bauer, Brett Bode, Catello Di Martino, Saurabh Jha,
  Chandra Narayanaswami, Daby Sow, Zbigniew T. Kalbarczyk, Ravishankar K. Iyer.
  "Story of Two GPUs: Characterizing the Resilience of Hopper H100 and
  Ampere A100 GPUs." SC '25.  arXiv:2503.11901   doi:10.1145/3712285.3759821

SAY "CALIBRATED TO", NEVER "VALIDATED AGAINST". These parameters are chosen so
the model reproduces published summary statistics. No claim is made that the
model predicts anything the telemetry confirms. That distinction is the whole
difference between an honest poster and one a resilience reviewer dismantles.

DO NOT USE the AVAIL_RSC1_* constants from the earlier draft here. Those come
from Kokolis et al. (HPCA 2025, arXiv:2410.21680), whose clusters are Meta's
RSC-1 (16K A100) and RSC-2 (8K A100). They are not Delta numbers.
"""

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# published Delta / DeltaAI figures
# --------------------------------------------------------------------------

DELTA = {
    "A100": {
        "mttr_hours": 0.88,          # mean node offline time for recovery
        "node_unavailability": 0.0060,
        "gpus_per_node": (4, 8),     # Delta A100 nodes come in both sizes
    },
    "H100": {
        "mttr_hours": 2.20,
        "node_unavailability": 0.0070,
        "gpus_per_node": (4,),       # Grace-Hopper nodes: 4 GPUs
    },
}

# TODO: CHECK THIS
DELTA_GPU_COUNT_UNVERIFIED = 1056


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------

def calibrate_rates(node_type="A100", step_hours=0.25):
    """Published availability + MTTR  ->  per-step failure and repair probs.

    A single component is a two-state chain with per-step failure prob a and
    repair prob b. Its stationary probability of being down is

        P(down) = a / (a + b)

    Mean time to repair is 1/b steps, so with a step of length step_hours:

        b = step_hours / MTTR_hours
        a = u b / (1 - u)        where u is the published unavailability

    step_hours must be < MTTR or b exceeds 1. Delta's A100 MTTR is 0.88 h, so
    0.25 h (15 min) is the natural choice and gives b ~ 0.28.
    """
    d = DELTA[node_type]
    mttr, u = d["mttr_hours"], d["node_unavailability"]
    if step_hours >= mttr:
        raise ValueError(
            f"step_hours={step_hours} must be < MTTR={mttr} h, else b > 1")
    b0 = step_hours / mttr
    a0 = u * b0 / (1.0 - u)
    return {"a0": a0, "b0": b0, "step_hours": step_hours,
            "mttr_hours": mttr, "node_unavailability": u,
            "node_type": node_type}


def mission_steps(job_hours, step_hours=0.25):
    """Job wall-clock -> mission horizon T."""
    return max(1, int(round(job_hours / step_hours)))


# --------------------------------------------------------------------------
# the payoff
# --------------------------------------------------------------------------

@dataclass
class CapacityPlan:
    p_T: float          # mission failure probability from the model
    r_f: float          # implied failure rate, per hour
    A: float            # steady-state availability
    n_target: int       # GPUs the SLA requires operational
    g: int              # GPUs per node
    n_nodes: int        # physical nodes needed
    n_prod: int         # physical GPUs needed
    overprovision: float  # fraction above n_target


def capacity_plan(p_T, T, step_hours, mttr_hours, n_target, g):
    """Turn a mission-failure probability into a provisioning decision.

    THE UNITS FIX. The earlier draft asserted r_f ~ p with no constant, but
    p_T is a dimensionless probability over a mission and r_f is a rate per
    unit time. Treat the mission as an exposure window of length T*step_hours
    and invert the survival function:

        1 - p_T = exp(-r_f * T * step_hours)
        r_f = -ln(1 - p_T) / (T * step_hours)          [per hour]

    For small p_T this reduces to r_f ~ p_T / (T * step_hours), which is the
    intended proportionality with the constant made explicit.

    Then the standard availability and provisioning identities:

        A       = 1 / (1 + r_f * MTTR)
        N_nodes = ceil(n_target / (g * A))
        N_prod  = g * N_nodes

    ------------------------------------------------------------------
    CONCEPTUAL TRAP -- resolve this before you put it on the poster.
    N_prod is driven by NODE availability. If the model's failure set F means
    "the job lost too much capacity", then p_T is a JOB failure probability,
    which is NOT the same quantity. Either
      (a) define F at the node level and keep this chain, or
      (b) keep F at the job level and state plainly that you are reporting the
          provisioning margin implied IF job-mission failure is used as the
          node disruption proxy.
    Silently substituting one for the other is exactly what a resilience
    reviewer will catch.
    ------------------------------------------------------------------
    """
    exposure = T * step_hours
    p_T = min(max(p_T, 0.0), 1.0 - 1e-15)
    r_f = -math.log(1.0 - p_T) / exposure
    A = 1.0 / (1.0 + r_f * mttr_hours)
    n_nodes = math.ceil(n_target / (g * A))
    n_prod = g * n_nodes
    return CapacityPlan(p_T=p_T, r_f=r_f, A=A, n_target=n_target, g=g,
                        n_nodes=n_nodes, n_prod=n_prod,
                        overprovision=n_prod / n_target - 1.0)


# --------------------------------------------------------------------------
# a Delta-shaped instance
# --------------------------------------------------------------------------

def delta_instance(N=8, node_type="A100", step_hours=0.25, job_hours=2.0,
                   big_nodes=4, gpus_big=8, gpus_small=4, request_frac=0.25,
                   gamma=0.30, eta=0.50):
    """Build a DynamicConfig-shaped dict for a heterogeneous Delta node pool.

    WHY HETEROGENEOUS CAPACITY IS PHYSICALLY REAL HERE: identical GPUs would
    give homogeneous c_i, which lumps the chain by failure count and makes the
    instance trivial (your own hardness dial). Delta's A100 partition contains
    BOTH 4-GPU and 8-GPU nodes, so a pool of N nodes has genuinely
    heterogeneous per-node capacity. The component is a NODE; c_i is its GPU
    count. That is the honest source of heterogeneity, not a modelling
    convenience.

    request_frac: the job asks for this fraction of nominal pool capacity.
    Tune it to put p_T in the rare-but-estimable band; the validation gate
    requires p_T < 0.01.
    """
    rates = calibrate_rates(node_type, step_hours)
    c = np.array([gpus_big] * big_nodes + [gpus_small] * (N - big_nodes),
                 dtype=float)
    c_nom = c.sum()
    return {
        "N": N,
        "T": mission_steps(job_hours, step_hours),
        "c": c,
        "c_min": float(np.floor(request_frac * c_nom)),
        "a0": np.full(N, rates["a0"]),
        "gamma": np.full(N, gamma),
        "b0": np.full(N, rates["b0"]),
        "eta": np.full(N, eta),
        "name": f"delta-{node_type}-N{N}",
        "_rates": rates,
        "_c_nom": c_nom,
    }