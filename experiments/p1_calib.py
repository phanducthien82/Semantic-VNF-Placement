"""Pilot P1 (after workload recalibration): objective weights and online behaviour of the
main methods.  Pilot seeds 900-905 are disjoint from the evaluation seeds."""
import argparse

from common import run_jobs
from online import run_online

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n_arr", type=int, default=600)
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(900, 906):
            for alg in ("Greedy", "Random"):
                jobs.append(dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr))
            for wH, wS in ((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)):
                jobs.append(dict(topo=topo, seed=seed, alg="Semantic-ILP", n_arr=a.n_arr, wH=wH, wS=wS))
            for alg in ("Semantic-GA", "GA"):
                jobs.append(dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr, wH=2.0, wS=1.0))
    run_jobs("p1_calib", jobs, run_online, workers=a.workers)
