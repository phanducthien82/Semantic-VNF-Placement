"""E9 -- Offered-load sweep (Section 4.4, Fig. 9).

Normal-state offered load rho_N in {0.5, 0.7, 0.9, 1.1, 1.3}; the burst state keeps the ratio of
Table 7 (rho_B = rho_N * 1.5 / 0.7).  10 seeds per point.
"""
import argparse

from common import RESULTS, run_jobs
from online import run_online

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_arr", type=int, default=1000)
    ap.add_argument("--algs", default="Semantic-ILP,Semantic-GA,GA,Greedy,DDPG,GCN-PPO")
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(a.seeds):
            for rho in (0.5, 0.7, 0.9, 1.1, 1.3):
                for alg in a.algs.split(","):
                    j = dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr, rho_n=rho, rho_b=round(rho * 1.5 / 0.7, 4))
                    if alg in ("DDPG", "GCN-PPO"):
                        j["agent_path"] = str(RESULTS / "drl" / f"{topo}_{alg}.pt")
                    jobs.append(j)
    run_jobs("e9_load", jobs, run_online, workers=a.workers)
