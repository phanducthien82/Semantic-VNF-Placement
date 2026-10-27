"""Pilot P0: calibration of the objective terms (headroom wH, policy-scarcity pricing wS).

Pilot seeds 900-905 are disjoint from the evaluation seeds (0-29) used in the paper.
"""
import argparse

from common import run_jobs
from online import run_online

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_arr", type=int, default=600)
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(900, 906):
            jobs.append(dict(topo=topo, seed=seed, alg="Greedy", n_arr=a.n_arr))
            for wH, wS in ((0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (2.0, 1.0)):
                jobs.append(dict(topo=topo, seed=seed, alg="Semantic-ILP", n_arr=a.n_arr, wH=wH, wS=wS))
    run_jobs("p0_weights", jobs, run_online, workers=a.workers)
