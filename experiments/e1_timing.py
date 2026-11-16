"""E1-T -- Decision times of all methods under identical, low-load conditions (Table 13, reply to R2.12).

The decision times recorded in the E1 runs were measured while other experiments ran in parallel
(Section 4.1), and not all methods ran under the same load.  For the timing comparison, seeds 0-4
of E1 are therefore re-run for every method with at most three concurrent runs on the six physical
cores and no other experiment running.  Only the timing fields of these runs are used.
"""
import argparse

from common import RESULTS, run_jobs
from online import run_online

ALGS = ("Semantic-ILP", "Semantic-GA", "GA", "Greedy", "Random", "DDPG", "GCN-PPO")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(a.seeds):
            for alg in ALGS:
                j = dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr, timing=1)
                if alg in ("DDPG", "GCN-PPO"):
                    j["agent_path"] = str(RESULTS / "drl" / f"{topo}_{alg}.pt")
                jobs.append(j)
    run_jobs("e1_timing", jobs, run_online, workers=a.workers)
