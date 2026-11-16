"""Event-driven online simulator (Section 4.1).

Every algorithm processes exactly the same arrival trace on exactly the same substrate for a
given seed (paired design).  For every arrival the orchestrator (i) refreshes the knowledge
view, (ii) evaluates the policy rules -> I_sem, (iii) calls the placement algorithm, and
(iv) passes the proposal to the algorithm-independent validator; only validated proposals
are deployed.  Energy and carbon are integrated exactly between events.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import ga as ga_mod
from . import ilp as ilp_mod
from .heuristics import greedy_placement, random_placement
from .model import DEFAULT_WEIGHTS, PI_TX_W_PER_MBPS, Instance, validate
from .routing import Router
from .rules import POLICIES, incompatibility, view_from_substrate


@dataclass
class SimConfig:
    algorithm: str = "Semantic-GA"
    warmup_frac: float = 0.2
    enabled: tuple = POLICIES            # policy types known to the semantic layer
    truth: tuple | None = None           # policy types enforced by the validator (default: enabled)
    semantic: bool = True                # False: policy-agnostic baseline (no I_sem)
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ga: ga_mod.GAConfig = field(default_factory=ga_mod.GAConfig)
    ilp_time_limit: float = 60.0
    ilp_gap: float = 1e-4
    k_paths: int = 5
    failures: dict | None = None          # {'at': 0.5, 'node_frac': 0.1, 'link_frac': 0.0, 'repair_s': 600}
    noise: dict | None = None             # imperfect knowledge (Section 4.9)
    unknown: str = "permit"               # unknown attributes: "permit" (sound, default) or "deny" (conservative)
    agent: object = None                  # trained DRL agent (DDPG / GCN-PPO)
    log_decisions: bool = False
    rdf_check: int = 0                    # every k-th arrival: run the full RDF pipeline and compare I_sem
    seed: int = 0


def _ga_cfg(base: ga_mod.GAConfig, alg: str, seed: int, k: int) -> ga_mod.GAConfig:
    d = dict(base.__dict__)
    d.update(ga_mod.VARIANTS[alg])
    d["seed"] = seed
    d["k_paths"] = k
    return ga_mod.GAConfig(**d)


class Simulator:
    def __init__(self, sub, trace, cfg: SimConfig):
        self.sub, self.trace, self.cfg = sub, trace, cfg
        self.rng = np.random.default_rng([cfg.seed, 33])
        self.archive = {}
        self.decisions = []
        self.rdf_checks = []

    # ------------------------------------------------------------------ helpers
    def _router(self):
        s = self.sub
        return Router(s.n, s.arcs, s.arc_prop + s.arc_tx, s.usable_arcs(), k=self.cfg.k_paths)

    def _power(self):
        s = self.sub
        act = s.n_vnfs > 0
        p_srv_idle = float((s.p_idle * act).sum())
        p_srv_dyn = float(((s.p_max - s.p_idle) * s.cpu_used / s.cpu_cap).sum())
        c_srv = float(((s.p_idle * act + (s.p_max - s.p_idle) * s.cpu_used / s.cpu_cap) * s.server_ci).sum())
        p_tx = PI_TX_W_PER_MBPS * float(s.arc_used.sum())
        return p_srv_idle, p_srv_dyn, p_tx, c_srv

    def _knowledge(self):
        kv = view_from_substrate(self.sub)
        if self.cfg.noise:
            from .noise import perturb
            return kv, perturb(kv, self.sub, self.cfg.noise, self.rng)
        return kv, kv

    # ------------------------------------------------------------------ one decision
    def decide(self, req, router, now):
        cfg, sub = self.cfg, self.sub
        kv_true, kv_obs = self._knowledge()
        t0 = time.perf_counter()
        if cfg.semantic:
            inc, per = incompatibility(req.vnfs, kv_obs, cfg.enabled, cfg.unknown)
        else:
            inc, per = np.zeros((req.n, sub.n), dtype=bool), {}
        t_sem = time.perf_counter() - t0
        if cfg.rdf_check and cfg.semantic and req.sfc_id % cfg.rdf_check == 0:
            from . import kb
            res_kb = kb.run_pipeline(sub, kv_obs, [req], now=now, enabled=cfg.enabled)
            self.rdf_checks.append(dict(equal=bool(np.array_equal(res_kb["isem"][req.sfc_id], inc)),
                                        **{k: v for k, v in res_kb.items() if k.startswith("t_")},
                                        abox=res_kb["abox_triples"], closure=res_kb["closure_triples"],
                                        tbox=res_kb["tbox_triples"], inconsistent=res_kb["inconsistent"],
                                        conforms=res_kb["validation"]["conforms"], t_compiled=t_sem))
        truth = cfg.truth if cfg.truth is not None else cfg.enabled
        inc_true, _ = incompatibility(req.vnfs, kv_true, truth)
        inst = Instance(req, sub, ~inc, weights=cfg.weights, kv=kv_obs)
        alg = cfg.algorithm
        extra = {}
        t1 = time.perf_counter()
        if alg == "Semantic-ILP" or alg == "ILP":
            res = ilp_mod.solve(inst, time_limit=cfg.ilp_time_limit, mip_gap=cfg.ilp_gap)
            extra = dict(ilp_status=res["status"], n_x=res["n_x"], n_f=res["n_f"], n_rows=res["n_rows"],
                         t_build=res["t_build"], cycles_removed=res["cycles_removed"])
        elif alg in ga_mod.VARIANTS:
            gcfg = _ga_cfg(cfg.ga, alg, int(self.rng.integers(2 ** 31 - 1)), cfg.k_paths)
            res = ga_mod.solve(inst, router, gcfg, archive=self.archive)
            extra = dict(n_eval=res.get("n_eval", 0))
        elif alg == "Greedy":
            res = greedy_placement(inst, router, use_semantic=True)
        elif alg == "Random":
            res = random_placement(inst, router, self.rng, use_semantic=True)
        elif alg in ("DDPG", "GCN-PPO"):
            res = cfg.agent.place(inst, router)
            extra = dict(t_pre=res.get("t_pre", 0.0), t_inf=res.get("t_inf", 0.0))
        else:
            raise KeyError(alg)
        t_alg = time.perf_counter() - t1
        servers, paths = res.get("servers"), res.get("paths")
        ok, reason = validate(inst, servers, paths, truth_allowed=~inc_true)
        if servers is None and reason == "no_solution":
            reason = res.get("status", "no_solution")
        n_pruned = int(inc.sum())
        rec = dict(sfc=req.sfc_id, tpl=req.template, t=now, ok=ok, reason=reason, t_sem=t_sem, t_alg=t_alg,
                   pruned_frac=n_pruned / (req.n * sub.n), space_log10=float(np.sum(np.log10(np.maximum(inst.cand.sum(axis=1), 1)))),
                   full_log10=float(req.n * np.log10(max(1, int(sub.node_up.sum())))),
                   burst=getattr(req, "burst", False), fn_pairs=int((inc_true & ~inc).sum()),
                   fp_pairs=int((inc & ~inc_true).sum()), **extra)
        emb = None
        if ok:
            emb = inst.to_embedding(servers, paths)
            ev = inst.evaluate(servers, paths)
            dist = router.dist()
            ends = inst.path_endpoints(servers)
            sp = sum(dist[u, v] for u, v in ends if u != v)
            rp = ev["lat_prop"] + ev["lat_tx"]
            cpu = np.array([v.cpu for v in req.vnfs], dtype=float)
            ci_srv = sub.server_ci[np.asarray(servers, dtype=int)]
            rec.update(latency=ev["latency"], lat_prop=ev["lat_prop"], lat_tx=ev["lat_tx"], lat_proc=ev["lat_proc"],
                       hops=ev["hops"], J=ev["J"], dP=ev["dP"], n_servers=ev["n_servers"], sla=req.sla_ms,
                       stretch=(rp / sp) if sp > 1e-9 else 1.0,
                       ci=float((cpu * ci_srv).sum() / max(cpu.sum(), 1e-9)))   # CPU-weighted CI of the hosts
        return emb, rec

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg, sub, trace = self.cfg, self.sub, self.trace
        sub.reset_state()
        M = len(trace)
        w_idx = int(cfg.warmup_frac * M)
        t_start = trace[w_idx].arrival
        t_end = trace[-1].arrival
        ev = []
        for k, r in enumerate(trace):
            heapq.heappush(ev, (r.arrival, 1, k, "arr", r))
        if cfg.failures:
            t_f = trace[int(cfg.failures.get("at", 0.5) * M)].arrival - 1e-6
            heapq.heappush(ev, (t_f, 0, -1, "fail", None))
        router = self._router()
        live = {}
        acc = Counter()
        reasons = Counter()
        per_tpl = {}
        tpl_acc = {}
        lat_sum = Counter()
        times = []
        e_idle = e_dyn = e_tx = carbon = 0.0
        act_int = util_int = mc_exp_int = mc_int = 0.0
        cpu_int = 0.0                                   # served core-seconds
        last_t = t_start
        interrupted = recovered = 0
        burst_acc = Counter()
        while ev:
            t, _, k, typ, payload = heapq.heappop(ev)
            if t > t_end and typ != "fail":
                break
            if t > last_t and t > t_start:
                t0_ = max(last_t, t_start)
                dt = t - t0_
                pi, pd, pt, cs = self._power()
                e_idle += pi * dt
                e_dyn += pd * dt
                e_tx += pt * dt
                carbon += (cs + pt * float(np.mean(sub.region_ci))) * dt
                act = sub.n_vnfs > 0
                act_int += act.sum() * dt
                cpu_int += float(sub.cpu_used.sum()) * dt
                util_int += (float((sub.cpu_used[act] / sub.cpu_cap[act]).mean()) if act.any() else 0.0) * dt
                mc_int += float(sub.mc_vnfs.sum()) * dt
                mc_exp_int += float(sub.mc_vnfs[sub.utilization() > 0.8].sum()) * dt
                last_t = t
            if typ == "dep":
                emb = live.pop(payload, None)
                if emb is not None and emb.expires <= t + 1e-9:
                    sub.release(emb)
                elif emb is not None:
                    live[payload] = emb
                continue
            if typ == "fail":
                self._fail(t, live, ev, router)
                router = self._router()
                for sid in list(live.keys()):
                    emb = live[sid]
                    if self._uses_down(emb):
                        sub.release(emb)
                        live.pop(sid)
                        interrupted += 1
                        req = emb.request
                        if not (sub.node_up[req.ingress] and sub.node_up[req.egress]):
                            continue
                        new, rec = self.decide(req, router, t)
                        if new is not None:
                            new.expires = emb.expires
                            sub.allocate(new)
                            live[sid] = new
                            recovered += 1
                continue
            if typ == "repair":
                sub.node_up[payload["nodes"]] = True
                sub.arc_up[payload["arcs"]] = True
                router = self._router()
                continue
            req = payload
            if not (sub.node_up[req.ingress] and sub.node_up[req.egress]):
                emb, rec = None, dict(sfc=req.sfc_id, tpl=req.template, ok=False, reason="endpoint_down",
                                     t_sem=0.0, t_alg=0.0, burst=getattr(req, "burst", False))
            else:
                emb, rec = self.decide(req, router, t)
            measured = k >= w_idx
            if measured:
                acc["arr"] += 1
                acc["acc"] += int(rec["ok"])
                reasons[rec["reason"]] += 1
                d = per_tpl.setdefault(req.template, [0, 0])
                d[0] += 1
                d[1] += int(rec["ok"])
                if rec["ok"]:   # per-slice host carbon intensity and latency/SLA of accepted SFCs
                    q = tpl_acc.setdefault(req.template, [0.0, 0.0, 0.0, 0])
                    q[0] += rec["ci"]
                    q[1] += rec["latency"]
                    q[2] += rec["latency"] / rec["sla"]
                    q[3] += 1
                if rec.get("burst"):
                    burst_acc["arr"] += 1
                    burst_acc["acc"] += int(rec["ok"])
                times.append((rec["t_sem"], rec["t_alg"], rec.get("t_pre", 0.0), rec.get("t_inf", 0.0)))
                for key in ("pruned_frac", "space_log10", "full_log10", "fn_pairs", "fp_pairs"):
                    if key in rec:
                        lat_sum[key] += rec[key]
                lat_sum["n_dec"] += 1
                lat_sum["cycles_removed"] += rec.get("cycles_removed", 0) or 0
                lat_sum["ilp_tl"] += int(rec.get("ilp_status") == "time_limit")
                if rec["ok"]:
                    for key in ("latency", "lat_prop", "lat_tx", "lat_proc", "hops", "J", "n_servers", "stretch"):
                        lat_sum[key] += rec[key]
                    lat_sum["lat_over_sla"] += rec["latency"] / rec["sla"]
                if cfg.log_decisions:
                    self.decisions.append(rec)
            if emb is not None:
                emb.expires = t + req.holding
                sub.allocate(emb)
                live[req.sfc_id] = emb
                heapq.heappush(ev, (emb.expires, 2, req.sfc_id, "dep", req.sfc_id))
        T = max(t_end - t_start, 1e-9)
        n_acc = max(acc["acc"], 1)
        tm = np.array(times) if times else np.zeros((1, 4))
        out = dict(arrivals=acc["arr"], accepted=acc["acc"], acceptance=acc["acc"] / max(acc["arr"], 1),
                   reasons=dict(reasons), per_template={k: v[1] / v[0] for k, v in per_tpl.items()},
                   per_template_ci={k: v[0] / v[3] for k, v in tpl_acc.items()},
                   per_template_latency={k: v[1] / v[3] for k, v in tpl_acc.items()},
                   per_template_lat_sla={k: v[2] / v[3] for k, v in tpl_acc.items()},
                   burst_acceptance=(burst_acc["acc"] / burst_acc["arr"]) if burst_acc["arr"] else None,
                   latency=lat_sum["latency"] / n_acc, lat_prop=lat_sum["lat_prop"] / n_acc,
                   lat_tx=lat_sum["lat_tx"] / n_acc, lat_proc=lat_sum["lat_proc"] / n_acc,
                   lat_over_sla=lat_sum["lat_over_sla"] / n_acc, hops=lat_sum["hops"] / n_acc,
                   stretch=lat_sum["stretch"] / n_acc, J=lat_sum["J"] / n_acc,
                   servers_per_sfc=lat_sum["n_servers"] / n_acc,
                   energy_kwh=(e_idle + e_dyn + e_tx) / 3.6e6, e_idle_kwh=e_idle / 3.6e6, e_dyn_kwh=e_dyn / 3.6e6,
                   e_tx_kwh=e_tx / 3.6e6, carbon_kg=carbon / 3.6e6 / 1000.0,
                   energy_per_acc_wh=(e_idle + e_dyn + e_tx) / 3600.0 / n_acc,
                   carbon_per_acc_g=carbon / 3.6e6 / n_acc,
                   active_servers=act_int / T, util_active=util_int / T,
                   served_core_h=cpu_int / 3600.0,
                   energy_per_core_h_wh=((e_idle + e_dyn + e_tx) / 3600.0) / max(cpu_int / 3600.0, 1e-9),
                   carbon_per_core_h_g=(carbon / 3.6e6) / max(cpu_int / 3600.0, 1e-9),
                   mc_exposure=(mc_exp_int / mc_int) if mc_int > 0 else 0.0,
                   t_sem_ms=1e3 * float(tm[:, 0].mean()), t_alg_ms=1e3 * float(tm[:, 1].mean()),
                   t_pre_ms=1e3 * float(tm[:, 2].mean()), t_inf_ms=1e3 * float(tm[:, 3].mean()),
                   t_dec_ms=1e3 * float((tm[:, 0] + tm[:, 1]).mean()),
                   t_dec_p95_ms=1e3 * float(np.percentile(tm[:, 0] + tm[:, 1], 95)),
                   pruned_frac=lat_sum["pruned_frac"] / max(lat_sum["n_dec"], 1),
                   space_log10=lat_sum["space_log10"] / max(lat_sum["n_dec"], 1),
                   full_log10=lat_sum["full_log10"] / max(lat_sum["n_dec"], 1),
                   fn_pairs=lat_sum["fn_pairs"] / max(lat_sum["n_dec"], 1),
                   fp_pairs=lat_sum["fp_pairs"] / max(lat_sum["n_dec"], 1),
                   interrupted=interrupted, recovered=recovered, sim_time_s=T,
                   cycles_removed=int(lat_sum["cycles_removed"]), ilp_time_limit_hits=int(lat_sum["ilp_tl"]))
        if self.rdf_checks:
            rc = self.rdf_checks
            out["rdf_checks"] = len(rc)
            out["rdf_equal"] = int(sum(r["equal"] for r in rc))
            for key in ("t_abox", "t_validate", "t_closure", "t_rules", "t_extract", "t_total", "abox", "closure",
                        "t_compiled"):
                out["rdf_" + key] = float(np.mean([r[key] for r in rc]))
            out["rdf_inconsistent"] = int(sum(r["inconsistent"] for r in rc))
            out["rdf_conforms"] = int(sum(r["conforms"] for r in rc))
        return out

    # ------------------------------------------------------------------ failures
    def _fail(self, t, live, ev, router):
        f = self.cfg.failures
        sub = self.sub
        frng = np.random.default_rng([self.cfg.seed, 44])
        nodes = np.array([], dtype=int)
        arcs = np.array([], dtype=int)
        if f.get("node_frac", 0) > 0:
            k = max(1, int(round(f["node_frac"] * sub.n)))
            nodes = frng.choice(sub.n, k, replace=False)
            sub.node_up[nodes] = False
        if f.get("link_frac", 0) > 0:
            k = max(1, int(round(f["link_frac"] * sub.topo.m)))
            links = frng.choice(sub.topo.m, k, replace=False)
            arcs = np.concatenate([2 * links, 2 * links + 1])
            sub.arc_up[arcs] = False
        if f.get("repair_s"):
            heapq.heappush(ev, (t + f["repair_s"], 0, -2, "repair", dict(nodes=nodes, arcs=arcs)))

    def _uses_down(self, emb) -> bool:
        sub = self.sub
        if any(not sub.node_up[j] for j in emb.servers):
            return True
        for p in emb.paths:
            for a in p:
                if not sub.arc_up[a] or not sub.node_up[sub.arcs[a, 0]] or not sub.node_up[sub.arcs[a, 1]]:
                    return True
        return False
