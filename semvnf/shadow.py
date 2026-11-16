"""Shadow evaluation on identical decision epochs (Sections 4.4, 4.7, 4.8).

The online state is driven by the exact Semantic-ILP.  At every ``stride``-th measured
arrival, a set of shadow solvers is evaluated on exactly the same instance (same
substrate state, same I_sem); their proposals are validated but not deployed.  This gives
paired, per-request comparisons with the exact optimum (optimality gap) without the state
divergence of independent online runs.
"""
from __future__ import annotations

import time

import numpy as np

from . import ga as ga_mod
from . import ilp as ilp_mod
from .heuristics import greedy_placement
from .model import Instance, validate
from .rules import incompatibility
from .sim import Simulator


class ShadowSimulator(Simulator):
    def __init__(self, sub, trace, cfg, shadows: list, stride: int = 5):
        super().__init__(sub, trace, cfg)
        self.shadows = shadows
        self.stride = stride
        self.records = []
        self._k = 0

    def decide(self, req, router, now):
        emb, rec = super().decide(req, router, now)
        self._k += 1
        w_idx = int(self.cfg.warmup_frac * len(self.trace))
        if req.sfc_id >= w_idx and req.sfc_id % self.stride == 0 and self.shadows:
            self._shadow(req, router, rec)
        return emb, rec

    def _shadow(self, req, router, rec):
        cfg, sub = self.cfg, self.sub
        kv_true, kv_obs = self._knowledge()
        inc, _ = incompatibility(req.vnfs, kv_obs, cfg.enabled)
        inst = Instance(req, sub, ~inc, weights=cfg.weights, kv=kv_obs)
        base = dict(sfc=req.sfc_id, tpl=req.template, n=req.n, ilp_ok=bool(rec["ok"]), ilp_J=rec.get("J"),
                    domain_log10=float(np.sum(np.log10(np.maximum(inst.cand.sum(axis=1), 1)))),
                    phys_log10=float(np.sum(np.log10(np.maximum(inst.cap_ok.sum(axis=1), 1)))),
                    pruned_pairs=int(inc.sum()))
        for sh in self.shadows:
            t0 = time.perf_counter()
            extra = {}
            if sh["kind"] == "ga":
                gcfg = ga_mod.GAConfig(**{**cfg.ga.__dict__, **sh.get("cfg", {}),
                                          "seed": int(self.rng.integers(2 ** 31 - 1))})
                rtr = router
                if gcfg.k_paths != router.k or sh.get("method", router.method) != router.method:
                    key = (gcfg.k_paths, sh.get("method", router.method))
                    cache = getattr(self, "_routers", {})
                    if key not in cache or cache[key][0] is not router:
                        from .routing import Router
                        cache[key] = (router, Router(sub.n, sub.arcs, sub.arc_prop + sub.arc_tx, sub.usable_arcs(),
                                                     k=gcfg.k_paths, method=key[1]))
                        self._routers = cache
                    rtr = cache[key][1]
                res = ga_mod.solve(inst, rtr, gcfg, archive=None, history=sh.get("history", False))
                extra = dict(n_eval=res.get("n_eval"), gens=res.get("gens"))
                if sh.get("history"):
                    extra["hist"] = res.get("hist")
            elif sh["kind"] == "greedy":
                res = greedy_placement(inst, router)
            elif sh["kind"] == "ilp":
                res = ilp_mod.solve(inst, time_limit=sh.get("time_limit", 60.0), use_pruning=sh.get("pruning", True),
                                    solver=sh.get("solver", "highs"))
                extra = dict(n_x=res["n_x"], n_f=res["n_f"], n_rows=res["n_rows"], status=res["status"],
                             t_build=res["t_build"], t_solve_only=res["t_solve"])
            else:
                raise KeyError(sh["kind"])
            dt = time.perf_counter() - t0
            ok, why = validate(inst, res.get("servers"), res.get("paths"))
            J = inst.evaluate(res["servers"], res["paths"])["J"] if ok else None
            gap = None
            if ok and base["ilp_ok"] and base["ilp_J"] is not None:
                gap = (J - base["ilp_J"]) / max(abs(base["ilp_J"]), 1e-9)
            self.records.append(dict(base, solver=sh["name"], ok=ok, reason=why, J=J, gap=gap, t=dt, **extra))
