"""Baseline heuristics (Section 4.1, Table 9).

* Random  -- every VNF on a uniformly random server of its domain with enough residual
             CPU/memory; virtual links on the latency-shortest path with enough residual
             bandwidth.
* Greedy  -- latency-greedy ("nearest feasible") placement: VNF i goes to the server of its
             domain minimising (path latency from the previous VNF + processing delay
             [+ latency to the egress for the last VNF]); routing as above.

Both receive the same semantic domains C_i as every other method ("semantic mask"); the
policy-agnostic variants used in Section 4.5 receive only the physical pre-check.
"""
from __future__ import annotations

import time

import numpy as np

from .model import Instance
from .routing import Router


def _route_chain(inst: Instance, router: Router, servers, res_bw):
    nodes = [inst.req.ingress] + list(servers) + [inst.req.egress]
    paths = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        p, _ = router.shortest(u, v, inst.req.bw, res_bw)
        if p is None:
            return None
        for a in p:
            res_bw[a] -= inst.req.bw
        paths.append(p)
    return paths


def random_placement(inst: Instance, router: Router, rng: np.random.Generator, use_semantic: bool = True):
    t0 = time.perf_counter()
    dom = inst.cand if use_semantic else inst.cap_ok
    res_cpu, res_mem, res_bw = inst.res_cpu.copy(), inst.res_mem.copy(), inst.res_bw.copy()
    servers = []
    for i in range(inst.n):
        d = np.flatnonzero(dom[i] & (res_cpu >= inst.cpu[i] - 1e-9) & (res_mem >= inst.mem[i] - 1e-9))
        if len(d) == 0:
            return dict(servers=None, paths=None, status="no_candidate", t_solve=time.perf_counter() - t0)
        j = int(d[rng.integers(len(d))])
        res_cpu[j] -= inst.cpu[i]
        res_mem[j] -= inst.mem[i]
        servers.append(j)
    paths = _route_chain(inst, router, servers, res_bw)
    st = "ok" if paths is not None else "no_path"
    return dict(servers=servers if paths else None, paths=paths, status=st, t_solve=time.perf_counter() - t0)


def greedy_placement(inst: Instance, router: Router, use_semantic: bool = True):
    t0 = time.perf_counter()
    dom = inst.cand if use_semantic else inst.cap_ok
    dist = router.dist()
    res_cpu, res_mem, res_bw = inst.res_cpu.copy(), inst.res_mem.copy(), inst.res_bw.copy()
    prev, servers, paths = inst.req.ingress, [], []
    for i in range(inst.n):
        d = np.flatnonzero(dom[i] & (res_cpu >= inst.cpu[i] - 1e-9) & (res_mem >= inst.mem[i] - 1e-9))
        if len(d) == 0:
            return dict(servers=None, paths=None, status="no_candidate", t_solve=time.perf_counter() - t0)
        score = dist[prev, d] + inst.d_proc[i, d]
        if i == inst.n - 1:
            score = score + dist[d, inst.req.egress]
        chosen = None
        for k in np.argsort(score, kind="stable"):
            j = int(d[k])
            if not np.isfinite(score[k]):
                break
            p, _ = router.shortest(prev, j, inst.req.bw, res_bw)
            if p is not None:
                chosen = (j, p)
                break
        if chosen is None:
            return dict(servers=None, paths=None, status="no_path", t_solve=time.perf_counter() - t0)
        j, p = chosen
        for a in p:
            res_bw[a] -= inst.req.bw
        res_cpu[j] -= inst.cpu[i]
        res_mem[j] -= inst.mem[i]
        servers.append(j)
        paths.append(p)
        prev = j
    p, _ = router.shortest(prev, inst.req.egress, inst.req.bw, res_bw)
    if p is None:
        return dict(servers=None, paths=None, status="no_path", t_solve=time.perf_counter() - t0)
    paths.append(p)
    return dict(servers=servers, paths=paths, status="ok", t_solve=time.perf_counter() - t0)
