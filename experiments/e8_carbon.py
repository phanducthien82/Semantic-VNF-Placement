"""E8 -- Energy-only versus carbon-aware objective (Section 4.10, reply to R2 comment 7).

Semantic-ILP and Semantic-GA with the carbon weight w_C in {0 (energy-only), 0.2 (default),
0.5 (carbon priority)}; all other weights fixed.  Reported: energy, operational carbon,
latency, acceptance ratio; and the effect on the delay-tolerant (mIoT) slice.
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
            for wc in (0.0, 0.2, 0.5):
                for alg in ("Semantic-ILP", "Semantic-GA"):
                    jobs.append(dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr, wC=wc))
    run_jobs("e8_carbon", jobs, run_online, workers=a.workers)
