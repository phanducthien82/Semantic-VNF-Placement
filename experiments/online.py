"""Job function shared by the online experiments (E1, E3, E4, E7-E10)."""
from __future__ import annotations

import time

from common import ROOT  # noqa: F401  (puts the package on sys.path)

from semvnf.ga import GAConfig
from semvnf.model import DEFAULT_WEIGHTS
from semvnf.rules import POLICIES
from semvnf.sim import SimConfig, Simulator
from semvnf.substrate import make_substrate
from semvnf.topology import load_topology
from semvnf.workload import WorkloadConfig, generate_trace

_AGENTS = {}


def _agent(alg, topo, path):
    key = (alg, topo, path)
    if key not in _AGENTS:
        from semvnf.drl import load_agent
        t0 = time.perf_counter()
        ag = load_agent(path)
        ag.t_load = time.perf_counter() - t0
        _AGENTS[key] = ag
    return _AGENTS[key]


def run_online(job: dict) -> dict:
    topo = load_topology(job["topo"], seed=job["seed"])
    sub = make_substrate(topo, job["seed"])
    wl = WorkloadConfig(n_arrivals=job.get("n_arr", 1000), rho_normal=job.get("rho_n", 0.7),
                        rho_burst=job.get("rho_b", 1.5))
    trace = generate_trace(sub, job["seed"], wl)
    w = dict(DEFAULT_WEIGHTS)
    for k in w:
        if k in job:
            w[k] = job[k]
    enabled = POLICIES if job.get("enabled", "all") == "all" else tuple(p for p in job["enabled"].split(",") if p)
    truth = None
    if job.get("truth") == "all":
        truth = POLICIES
    gcfg = GAConfig(pop=job.get("pop", 50), gens=job.get("gens", 100), k_paths=job.get("k", 5),
                    penalty=job.get("penalty", "segregated"), omega=job.get("omega", 1.0))
    failures = None
    if job.get("fail_nodes", 0) or job.get("fail_links", 0):
        failures = dict(at=0.5, node_frac=job.get("fail_nodes", 0.0), link_frac=job.get("fail_links", 0.0),
                        repair_s=job.get("repair_s", 600.0))
    noise = None
    if job.get("noise", 0):
        noise = dict(rate=job["noise"], kind=job.get("noise_kind", "mixed"))
    agent = None
    if job["alg"] in ("DDPG", "GCN-PPO"):
        agent = _agent(job["alg"], job["topo"], job["agent_path"])
    cfg = SimConfig(algorithm=job["alg"], enabled=enabled, truth=truth, semantic=job.get("semantic", True),
                    weights=w, ga=gcfg, k_paths=job.get("k", 5), failures=failures, noise=noise, agent=agent,
                    unknown=job.get("unknown", "permit"),
                    ilp_time_limit=job.get("ilp_tl", 60.0), seed=job["seed"],
                    log_decisions=job.get("log", False), rdf_check=job.get("rdf_check", 0))
    sim = Simulator(sub, trace, cfg)
    if agent is not None:
        agent.reset_episode_state() if hasattr(agent, "reset_episode_state") else None
    out = sim.run()
    if agent is not None:
        out["t_load_s"] = getattr(agent, "t_load", None)
        out["t_train_s"] = getattr(agent, "train_time_s", None)
    if job.get("log", False):
        out["decisions"] = sim.decisions
    return out
