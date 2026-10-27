"""E4 -- Policy ablation (Section 4.5, reply to R1 comment 15 and R2 comment 1).

Configurations (Semantic-ILP, exact, so that the effect of a policy is not confounded by
heuristic noise):
  all          all seven policy types known and enforced;
  -<p>         leave-one-out: policy type p removed from the policy set (known and enforced);
  none         no policy type;
  agnostic     all seven enforced by the admission validator but unknown to the optimiser
               (semantic layer switched off) -> measures the value of semantic pruning.
Computational cost per policy type (rule-evaluation time, pruned pairs) is measured with the
full RDF pipeline in E0/E5.
"""
import argparse

from common import run_jobs
from online import run_online

from semvnf.rules import POLICIES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    cfgs = [("all", dict())]
    for p in POLICIES:
        cfgs.append((f"-{p}", dict(enabled=",".join(q for q in POLICIES if q != p))))
    cfgs.append(("none", dict(enabled="")))
    cfgs.append(("agnostic", dict(semantic=False, truth="all")))
    jobs = []
    for topo in ("euro28", "germany50"):
        for seed in range(a.seeds):
            for name, extra in cfgs:
                for alg in ("Semantic-ILP",) + (("Greedy",) if name in ("all", "agnostic") else ()):
                    jobs.append(dict(topo=topo, seed=seed, alg=alg, n_arr=a.n_arr, cfg=name, **extra))
    run_jobs("e4_policies", jobs, run_online, workers=a.workers)
