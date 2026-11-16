"""E5 -- Scalability on Barabasi-Albert substrates (Section 4.6, reply to R2 comments 13-14
and R1 comment 10).

Generation procedure (fully reproducible):
  * topology   BA(N, m=2) with networkx, graph seed derived from the run seed; link
               propagation latency U(1, 5) ms; core sites = top-20% degree nodes; regions =
               hop-distance Voronoi cells around round(N/12) hubs (carbon intensity per region);
  * attributes Table 7 (same distributions as for the SNDlib topologies);
  * load       requests of the reference mix are embedded with Greedy (no departures) until
               the mean CPU utilisation reaches 50 %; this creates tenants, exclusive servers
               and hot servers so that every policy type is active;
  * test set   the next R requests of the trace are solved on the resulting snapshot
               (without being deployed) by every method.
Per seed we also run the full RDF pipeline once on the snapshot (ABox + all R test SFCs).
"""
import argparse
import time

import numpy as np
from common import run_jobs

from semvnf import ga as ga_mod
from semvnf import ilp as ilp_mod
from semvnf import kb
from semvnf.heuristics import greedy_placement
from semvnf.model import DEFAULT_WEIGHTS, Instance, validate
from semvnf.routing import Router
from semvnf.rules import incompatibility, view_from_substrate
from semvnf.substrate import make_substrate
from semvnf.topology import load_topology
from semvnf.workload import WorkloadConfig, generate_trace


def preload(sub, trace, target_util=0.5):
    router = Router(sub.n, sub.arcs, sub.arc_prop + sub.arc_tx, sub.usable_arcs(), k=5)
    k = 0
    while sub.cpu_used.sum() / sub.cpu_cap.sum() < target_util and k < len(trace):
        req = trace[k]
        kv = view_from_substrate(sub)
        inc, _ = incompatibility(req.vnfs, kv)
        inst = Instance(req, sub, ~inc)
        res = greedy_placement(inst, router)
        ok, _ = validate(inst, res.get("servers"), res.get("paths"))
        if ok:
            sub.allocate(inst.to_embedding(res["servers"], res["paths"]))
        k += 1
    return k, router


def run(job):
    N, seed, R = job["N"], job["seed"], job["R"]
    topo = load_topology(f"ba-{N}", seed=seed)
    sub = make_substrate(topo, seed)
    trace = generate_trace(sub, seed, WorkloadConfig(n_arrivals=40 * N))
    t0 = time.perf_counter()
    k, router = preload(sub, trace)
    t_pre = time.perf_counter() - t0
    tests = trace[k:k + R]
    out = dict(N=N, m=int(topo.m), preload_requests=k, util=float(sub.cpu_used.sum() / sub.cpu_cap.sum()),
               t_preload=t_pre)
    kv = view_from_substrate(sub)
    if job.get("rdf", True):
        res = kb.run_pipeline(sub, kv, tests, now=0.0)
        out.update(rdf_abox=res["abox_triples"], rdf_closure=res["closure_triples"], rdf_tbox=res["tbox_triples"],
                   rdf_rule_time=res["rule_time"], rdf_per_policy=res["per_policy"],
                   **{"rdf_" + k2: v for k2, v in res.items() if k2.startswith("t_")})
        eq = [bool(np.array_equal(res["isem"][r.sfc_id], incompatibility(r.vnfs, kv)[0])) for r in tests]
        out["rdf_equal"] = int(sum(eq))
    t0 = time.perf_counter()
    _ = router.dist()
    out["t_apsp"] = time.perf_counter() - t0
    w = dict(DEFAULT_WEIGHTS)
    for key in ("wH", "wS"):
        if key in job:
            w[key] = job[key]
    per = []
    for req in tests:
        rec = dict(n=req.n, tpl=req.template)
        t0 = time.perf_counter()
        inc, _ = incompatibility(req.vnfs, kv)
        rec["t_rules_ms"] = 1e3 * (time.perf_counter() - t0)
        inst = Instance(req, sub, ~inc, weights=w, kv=kv)
        rec["domain_log10"] = float(np.sum(np.log10(np.maximum(inst.cand.sum(axis=1), 1))))
        rec["phys_log10"] = float(np.sum(np.log10(np.maximum(inst.cap_ok.sum(axis=1), 1))))
        rec["full_log10"] = float(req.n * np.log10(sub.n))
        rec["pruned_frac"] = float(inc.sum() / inc.size)
        if job.get("ilp", True):
            r = ilp_mod.solve(inst, time_limit=job.get("ilp_tl", 300.0))
            ok, _ = validate(inst, r["servers"], r["paths"])
            rec.update(ilp_t=r["t_build"] + r["t_solve"], ilp_status=r["status"], ilp_ok=ok,
                       ilp_J=(inst.evaluate(r["servers"], r["paths"])["J"] if ok else None),
                       ilp_nx=r["n_x"], ilp_nf=r["n_f"], ilp_rows=r["n_rows"], ilp_gap=r["gap"])
        for name in ("Semantic-GA", "GA"):
            cfg = ga_mod.GAConfig(**{**ga_mod.VARIANTS[name], "seed": seed * 1000 + req.sfc_id})
            t0 = time.perf_counter()
            r = ga_mod.solve(inst, router, cfg)
            dt = time.perf_counter() - t0
            ok, _ = validate(inst, r.get("servers"), r.get("paths"))
            rec[f"{name}_t"] = dt
            rec[f"{name}_ok"] = ok
            rec[f"{name}_J"] = inst.evaluate(r["servers"], r["paths"])["J"] if ok else None
        t0 = time.perf_counter()
        r = greedy_placement(inst, router)
        rec["Greedy_t"] = time.perf_counter() - t0
        ok, _ = validate(inst, r.get("servers"), r.get("paths"))
        rec["Greedy_ok"] = ok
        rec["Greedy_J"] = inst.evaluate(r["servers"], r["paths"])["J"] if ok else None
        per.append(rec)
    out["requests"] = per
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--sizes", default="50,100,200,500,1000")
    ap.add_argument("--wH", type=float, default=DEFAULT_WEIGHTS["wH"])
    ap.add_argument("--wS", type=float, default=DEFAULT_WEIGHTS["wS"])
    a = ap.parse_args()
    jobs = [dict(N=int(N), seed=s, R=a.R, wH=a.wH, wS=a.wS) for N in a.sizes.split(",") for s in range(a.seeds)]
    run_jobs("e5_scalability", jobs, run, workers=a.workers)
