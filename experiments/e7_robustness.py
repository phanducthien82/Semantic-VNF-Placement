"""E7 -- Imperfect knowledge (Section 4.9, reply to R2 comment 6).

The orchestrator reasons on an observed knowledge view perturbed with rate r in
{0, 0.02, 0.05, 0.10, 0.20} (stale telemetry, unvalidated inventory, missing carbon
observations and wrong assertions; semvnf/noise.py); the admission validator uses the ground
truth.  Reported: false-positive / false-negative pairs of I_sem, proposals rejected for a
policy violation, acceptance ratio and objective.
"""
import argparse

from common import run_jobs
from online import run_online

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(a.seeds):
            for rate in (0.0, 0.02, 0.05, 0.10, 0.20):
                for kind in (("mixed",) if rate == 0 else ("mixed", "stale", "wrong")):
                    jobs.append(dict(topo=topo, seed=seed, alg="Semantic-ILP", n_arr=a.n_arr, noise=rate,
                                     noise_kind=kind))
    run_jobs("e7_robustness", jobs, run_online, workers=a.workers)
