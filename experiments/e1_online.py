"""E1 -- Main online comparison on Euro28 and Germany50 (Section 4.4, Table 12, Figs. 8-10).

30 evaluation seeds (0-29); 1000 SFC arrivals per run (first 20 % = warm-up); MMPP traffic
(rho_N = 0.7, rho_B = 1.5, burst share 10 %).  Scenario 'fail' additionally fails 10 % of
the nodes at mid-run for 600 s (affected SFCs are re-embedded by the same algorithm).
The Semantic-ILP runs also execute the full RDF/OWL pipeline every 50th arrival and compare
its incompatibility set with the compiled rules (experiment E0).
"""
import argparse

from common import RESULTS, run_jobs
from online import run_online

ALGS = ("Semantic-ILP", "Semantic-GA", "GA", "Greedy", "Random", "DDPG", "GCN-PPO")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--algs", default=",".join(ALGS))
    ap.add_argument("--scenario", default="normal", choices=["normal", "fail"])
    ap.add_argument("--n_arr", type=int, default=1000)
    ap.add_argument("--topos", default="euro28,germany50")
    a = ap.parse_args()
    jobs = []
    for topo in a.topos.split(","):
        for seed in range(a.seeds):
            for alg in a.algs.split(","):
                j = dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr)
                if a.scenario == "fail":
                    j.update(fail_nodes=0.1, repair_s=600.0)
                if alg in ("DDPG", "GCN-PPO"):
                    j["agent_path"] = str(RESULTS / "drl" / f"{topo}_{alg}.pt")
                if alg == "Semantic-ILP" and a.scenario == "normal":
                    j["rdf_check"] = 50
                if seed < 5:
                    j["log"] = True
                jobs.append(j)
    run_jobs("e1_online" if a.scenario == "normal" else "e1_fail", jobs, run_online, workers=a.workers)
