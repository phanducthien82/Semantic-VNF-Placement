"""E7b -- Imperfect knowledge with the conservative compilation mode (Section 4.9, reply to R2 comment 6).

Same error model as E7, but the placement snapshot is compiled with unknown="deny": a pair whose
policy-relevant attribute is unknown (stale resource state, unvalidated inventory, missing carbon
observation) is treated as incompatible.  Proposition 1 predicts that this removes the false
negatives caused by unknown attributes (compliance) at the price of false positives (lost options),
whereas the default mode (unknown="permit", E7) has no false positives from unknown attributes.
"""
import argparse

from common import run_jobs
from online import run_online

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(a.seeds):
            for rate in (0.05, 0.20):
                for kind in ("stale", "mixed"):
                    jobs.append(dict(topo=topo, seed=seed, alg="Semantic-ILP", n_arr=a.n_arr, noise=rate,
                                     noise_kind=kind, unknown="deny"))
    run_jobs("e7b_conservative", jobs, run_online, workers=a.workers)
