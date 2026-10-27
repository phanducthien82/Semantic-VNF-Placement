"""Train the DRL baselines (DDPG, GCN-PPO) per SNDlib topology on training seeds 1000+.

Training seeds are disjoint from the evaluation seeds 0-29.  The learning curve (acceptance
ratio and mean objective per training trace) and the wall-clock training time are stored in
results/drl/<topo>_<alg>.json; the checkpoint in results/drl/<topo>_<alg>.pt.
"""
import argparse
import json
import time

from common import RESULTS

from semvnf.drl import DDPGAgent, PPOAgent, save_agent
from semvnf.model import DEFAULT_WEIGHTS
from semvnf.sim import SimConfig, Simulator
from semvnf.substrate import make_substrate
from semvnf.topology import load_topology
from semvnf.workload import WorkloadConfig, generate_trace


def train(topo_name: str, alg: str, n_traces: int, n_arr: int, weights: dict, seed0: int = 1000):
    topo = load_topology(topo_name)
    sub0 = make_substrate(topo, seed0)
    agent = DDPGAgent(topo.n, seed=7) if alg == "DDPG" else PPOAgent(sub0, seed=7)
    agent.training = True
    curve = []
    t0 = time.perf_counter()
    for k in range(n_traces):
        seed = seed0 + k
        sub = make_substrate(topo, seed)
        trace = generate_trace(sub, seed, WorkloadConfig(n_arrivals=n_arr))
        sim = Simulator(sub, trace, SimConfig(algorithm=alg, agent=agent, seed=seed, weights=weights,
                                              warmup_frac=0.0))
        out = sim.run()
        curve.append(dict(trace=k, seed=seed, acceptance=out["acceptance"], J=out["J"],
                          elapsed_s=time.perf_counter() - t0))
        print(f"{topo_name} {alg} trace {k}: acc={out['acceptance']:.3f} J={out['J']:.3f} "
              f"t={time.perf_counter() - t0:.0f}s", flush=True)
    agent.train_time_s = time.perf_counter() - t0
    agent.training = False
    out_dir = RESULTS / "drl"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_agent(agent, out_dir / f"{topo_name}_{alg}.pt")
    (out_dir / f"{topo_name}_{alg}.json").write_text(json.dumps(dict(
        topo=topo_name, alg=alg, n_traces=n_traces, n_arr=n_arr, train_time_s=agent.train_time_s,
        weights=weights, curve=curve), indent=1))
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="euro28")
    ap.add_argument("--alg", default="DDPG")
    ap.add_argument("--traces", type=int, default=12)
    ap.add_argument("--n_arr", type=int, default=1000)
    ap.add_argument("--wH", type=float, default=DEFAULT_WEIGHTS["wH"])
    ap.add_argument("--wS", type=float, default=DEFAULT_WEIGHTS["wS"])
    a = ap.parse_args()
    w = dict(DEFAULT_WEIGHTS, wH=a.wH, wS=a.wS)
    train(a.topo, a.alg, a.traces, a.n_arr, w)
