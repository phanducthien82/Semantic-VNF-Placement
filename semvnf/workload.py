"""Workload model (Section 4.1, Table 7).

SFC requests arrive according to a two-state Markov-modulated Poisson process (MMPP):
state N (normal, offered load rho_N) and state B (burst, offered load rho_B), with
exponential sojourn times.  The arrival rate of a state follows from its offered load:

    lambda = rho * C_total / (E[CPU per SFC] * E[holding time])

so that the workload scales linearly with the total CPU capacity of the substrate
(scaling law used for every topology size).  Holding times are exponential.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import TEMPLATES, mean_cpu_per_sfc, vnf_specs
from .model import Request


@dataclass
class WorkloadConfig:
    n_arrivals: int = 1000
    rho_normal: float = 0.7
    rho_burst: float = 1.5
    burst_share: float = 0.1          # stationary probability of the burst state
    burst_mean_s: float = 120.0       # mean sojourn time in the burst state
    holding_mean_s: float = 120.0     # 1/mu
    template_probs: tuple | None = None


def mmpp_rates(cfg: WorkloadConfig, cpu_total: float) -> tuple[float, float]:
    base = cpu_total / (mean_cpu_per_sfc() * cfg.holding_mean_s)
    return cfg.rho_normal * base, cfg.rho_burst * base


def generate_trace(sub, seed: int, cfg: WorkloadConfig) -> list[Request]:
    """Arrival trace for one seed (identical for every algorithm: paired design)."""
    rng = np.random.default_rng([seed, 22])
    lam_n, lam_b = mmpp_rates(cfg, float(sub.cpu_cap.sum()))
    normal_mean = cfg.burst_mean_s * (1 - cfg.burst_share) / cfg.burst_share
    probs = np.array(cfg.template_probs or [t.prob for t in TEMPLATES], dtype=float)
    probs /= probs.sum()
    region = sub.topo.region
    by_region = {r: np.flatnonzero(region == r) for r in np.unique(region)}
    t, state = 0.0, 0
    seg_end = rng.exponential(normal_mean)
    out = []
    sid = 0
    while len(out) < cfg.n_arrivals:
        lam = lam_n if state == 0 else lam_b
        dt = rng.exponential(1.0 / lam)
        if t + dt > seg_end:                          # state change before next arrival
            t = seg_end
            state = 1 - state
            seg_end = t + rng.exponential(cfg.burst_mean_s if state == 1 else normal_mean)
            continue
        t += dt
        tpl = TEMPLATES[int(rng.choice(len(TEMPLATES), p=probs))]
        ing = int(rng.integers(sub.n))
        if tpl.local:                                  # local breakout at the ingress MEC site
            egr = ing
        else:
            egr = int(rng.integers(sub.n))
        bw = float(np.round(rng.uniform(*tpl.bw)))
        hold = float(rng.exponential(cfg.holding_mean_s))
        out.append(Request(sfc_id=sid, template=tpl.name, ingress=ing, egress=egr, vnfs=vnf_specs(tpl),
                           bw=bw, sla_ms=tpl.sla_ms, arrival=t, holding=hold))
        out[-1].burst = bool(state == 1)
        sid += 1
    return out
