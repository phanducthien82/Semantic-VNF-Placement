"""E4b -- How often each policy type fires on the decision epochs of Semantic-ILP (Section 4.5).

For every arrival of Semantic-ILP runs (seeds 0-4) we record, per policy type, the share of
(VNF, server) pairs it prunes, the share it prunes *alone* (no other policy prunes the pair) and
whether it empties the candidate domain of some VNF of the request (blocking).
"""
import argparse

import numpy as np
from common import run_jobs

from semvnf.ga import GAConfig
from semvnf.model import DEFAULT_WEIGHTS
from semvnf.rules import POLICIES
from semvnf.sim import SimConfig, Simulator
from semvnf.substrate import make_substrate
from semvnf.topology import load_topology
from semvnf.workload import WorkloadConfig, generate_trace


class RecordingSimulator(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.acc = {p: dict(pairs=0.0, unique=0.0, blocks=0) for p in POLICIES + ("any",)}
        self.n_dec = 0
        self.n_pairs = 0

    def decide(self, req, router, now):
        from semvnf.rules import evaluate_policies
        kv_true, kv_obs = self._knowledge()
        per = evaluate_policies(req.vnfs, kv_obs, self.cfg.enabled)
        stack = np.stack([per[p] for p in POLICIES])            # (P, n, N)
        count = stack.sum(axis=0)
        cap_ok = ((np.array([v.cpu for v in req.vnfs])[:, None] <= self.sub.res_cpu()[None, :])
                  & self.sub.node_up[None, :])
        self.n_dec += 1
        self.n_pairs += stack.shape[1] * stack.shape[2]
        for k, p in enumerate(POLICIES):
            m = stack[k]
            self.acc[p]["pairs"] += float(m.sum())
            self.acc[p]["unique"] += float((m & (count == 1)).sum())
            others = (count - m.astype(int)) > 0
            # p blocks the request if some VNF has candidates without p but none with p
            dom_without = cap_ok & ~others
            dom_with = dom_without & ~m
            if np.any(dom_without.any(axis=1) & ~dom_with.any(axis=1)):
                self.acc[p]["blocks"] += 1
        # all policies together: pruned pairs, and requests whose domain is emptied by the policies
        anyp = count > 0
        self.acc["any"]["pairs"] += float(anyp.sum())
        self.acc["any"]["unique"] += float(anyp.sum())
        if np.any(cap_ok.any(axis=1) & ~(cap_ok & ~anyp).any(axis=1)):
            self.acc["any"]["blocks"] += 1
        return super().decide(req, router, now)


def run(job):
    topo = load_topology(job["topo"])
    sub = make_substrate(topo, job["seed"])
    trace = generate_trace(sub, job["seed"], WorkloadConfig(n_arrivals=job["n_arr"]))
    sim = RecordingSimulator(sub, trace, SimConfig(algorithm="Semantic-ILP", weights=dict(DEFAULT_WEIGHTS),
                                                   ga=GAConfig(), seed=job["seed"]))
    out = sim.run()
    res = dict(acceptance=out["acceptance"], n_dec=sim.n_dec, n_pairs=sim.n_pairs)
    for p, d in sim.acc.items():
        res[f"{p}/pairs"] = d["pairs"] / max(sim.n_pairs, 1)
        res[f"{p}/unique"] = d["unique"] / max(sim.n_pairs, 1)
        res[f"{p}/blocks"] = d["blocks"] / max(sim.n_dec, 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_arr", type=int, default=1000)
    a = ap.parse_args()
    jobs = [dict(topo=t, seed=s, n_arr=a.n_arr) for t in ("euro28", "germany50") for s in range(a.seeds)]
    run_jobs("e4b_policy_shares", jobs, run, workers=a.workers)
