"""E2/E3/E6/E10 -- Shadow evaluation on identical decision epochs (Sections 4.5, 4.7, 4.8).

The online state is driven by Semantic-ILP; every 5th measured arrival, the following solvers
are evaluated on exactly the same instance and compared with the exact optimum:

  E3 GA ablation (equal budget 50 x 100 evaluations): GA, GA+P, GA+W, GA+P+W, Semantic-GA;
  E10 path-set size K in {1, 2, 3, 5, 8} (LFA) and Yen K=5 for Semantic-GA;
  E6 static penalty Omega in {0.01, 0.1, 1, 10, 100} versus the feasibility-first rule;
  E2 Semantic-ILP without pruning (policies as explicit rows) and Gurobi cross-check.
"""
import argparse

from common import ROOT, run_jobs  # noqa: F401

from semvnf.ga import VARIANTS, GAConfig
from semvnf.model import DEFAULT_WEIGHTS
from semvnf.shadow import ShadowSimulator
from semvnf.sim import SimConfig
from semvnf.substrate import make_substrate
from semvnf.topology import load_topology
from semvnf.workload import WorkloadConfig, generate_trace


def shadows():
    sh = []
    for name, v in VARIANTS.items():
        sh.append(dict(name=name, kind="ga", cfg=dict(v), history=(name in ("GA", "GA+P", "GA+W", "GA+P+W", "Semantic-GA"))))
    for k in (1, 2, 3, 8):
        sh.append(dict(name=f"SGA-K{k}", kind="ga", cfg=dict(VARIANTS["Semantic-GA"], k_paths=k)))
    sh.append(dict(name="SGA-Yen5", kind="ga", cfg=dict(VARIANTS["Semantic-GA"]), method="yen"))
    for om in (0.01, 0.1, 1.0, 10.0, 100.0):
        sh.append(dict(name=f"GA-static-{om:g}", kind="ga", cfg=dict(VARIANTS["GA"], penalty="static", omega=om)))
    sh.append(dict(name="GA-segregated", kind="ga", cfg=dict(VARIANTS["GA"])))
    sh.append(dict(name="ILP-nopruning", kind="ilp", pruning=False))
    sh.append(dict(name="ILP-gurobi", kind="ilp", solver="gurobi"))
    sh.append(dict(name="Greedy", kind="greedy"))
    return sh


def run(job):
    topo = load_topology(job["topo"], seed=job["seed"])
    sub = make_substrate(topo, job["seed"])
    trace = generate_trace(sub, job["seed"], WorkloadConfig(n_arrivals=job["n_arr"]))
    cfg = SimConfig(algorithm="Semantic-ILP", weights=dict(DEFAULT_WEIGHTS), ga=GAConfig(), seed=job["seed"])
    sim = ShadowSimulator(sub, trace, cfg, shadows(), stride=job.get("stride", 5))
    out = sim.run()
    recs = []
    for r in sim.records:
        r = dict(r)
        if "hist" in r:
            if r["hist"] and r["sfc"] % 20 == 0:
                r["hist"] = [[int(a), float(b), float(c), float(d)] for a, b, c, d in r["hist"]]
            else:
                r.pop("hist")
        recs.append(r)
    return dict(driver=dict(acceptance=out["acceptance"]), records=recs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    jobs = [dict(topo=t, seed=s, n_arr=a.n_arr) for t in ("euro28", "germany50") for s in range(a.seeds)]
    run_jobs("e2_shadow", jobs, run, workers=a.workers)
