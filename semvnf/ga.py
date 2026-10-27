"""Semantic-GA and its ablation variants (Section 3.5, Table 11).

Chromosome C = [X | F]:
  X_i in D_i       server of VNF i (D_i = C_i with semantic pruning; otherwise all operational
                   servers with enough residual CPU/memory for VNF i);
  F_e in {0..K-1}  index of the path of virtual link e in the candidate set P_K(end_e, end_{e+1})
                   of the endpoints *given X* (ingress, X_1, ..., X_n, egress), see routing.py.

Decoding is deterministic: path(e) = P_K(end_e, end_{e+1})[F_e mod |P_K|]; every decoded
chromosome therefore satisfies endpoint consistency and flow conservation (Proposition 4).
Node/link capacity, the SLA and (for variants without pruning) the policies are handled by the
feasibility-first fitness (Proposition 3)

    Phi(C) = J(C)              if V(C) = 0,
    Phi(C) = J_ub + V(C)       otherwise,

so that every feasible chromosome is strictly better than every infeasible one.  The static
penalty Phi = J + Omega V is available for the sensitivity study (Section 4.8).
The population is evaluated in a vectorised way (numpy), which is what makes sub-second
decisions possible in Python.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .model import Instance
from .routing import Router


@dataclass
class GAConfig:
    pop: int = 50
    gens: int = 100
    pc: float = 0.8
    pm: float = 0.1
    tour: int = 3
    elite: int = 2
    k_paths: int = 5
    prune: bool = True           # semantic domain restriction (initialisation + mutation)
    warm: bool = True            # warm start (greedy, consolidation and archive seeds)
    local_search: bool = True    # memetic refinement of the best individual
    ls_share: float = 0.1        # share of the evaluation budget reserved for local search
    penalty: str = "segregated"  # 'segregated' | 'static'
    omega: float = 1.0           # static penalty weight (penalty='static')
    w_node: float = 1.0
    w_link: float = 1.0
    w_sla: float = 1.0
    w_pol: float = 1.0
    seed: int = 0


VARIANTS = {
    "GA":          dict(prune=False, warm=False, local_search=False),
    "GA+P":        dict(prune=True, warm=False, local_search=False),
    "GA+W":        dict(prune=False, warm=True, local_search=False),
    "GA+P+W":      dict(prune=True, warm=True, local_search=False),
    "Semantic-GA": dict(prune=True, warm=True, local_search=True),
}


class GAEngine:
    """GA bound to one decision epoch (instance + router)."""

    def __init__(self, inst: Instance, router: Router, cfg: GAConfig, archive: dict | None = None):
        self.inst, self.router, self.cfg = inst, router, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = inst.n
        r = inst.req
        self.src, self.dst, self.bw = r.ingress, r.egress, r.bw
        self.dom_sem = [np.flatnonzero(inst.cand[i]) for i in range(self.n)]
        self.dom_phys = [np.flatnonzero(inst.cap_ok[i]) for i in range(self.n)]
        self.dom = self.dom_sem if cfg.prune else self.dom_phys
        self.J_ub = inst.J_upper_bound()
        self.archive = archive
        self.pcache = {}
        self.n_eval = 0
        self.N, self.A = inst.N, inst.sub.n_arcs
        self.ar_n = np.arange(self.n)

    # ------------------------------------------------------------------ path lookup
    def _path(self, u, v, k):
        key = (u, v, k)
        hit = self.pcache.get(key)
        if hit is None:
            ps = self.router.paths(u, v)
            if not ps:
                hit = (None, 0.0, 0.0)
            else:
                arcs, lat = ps[k % len(ps)]
                arr = np.asarray(arcs, dtype=np.int64)
                hit = (arr, lat, float(self.inst.c_f[arr].sum()) if len(arr) else 0.0)
            self.pcache[key] = hit
        return hit

    # ------------------------------------------------------------------ vectorised evaluation
    def evaluate(self, X: np.ndarray, F: np.ndarray):
        """X: (P, n), F: (P, n+1) -> phi, J, V, latency (each (P,))."""
        inst, cfg = self.inst, self.cfg
        P, n, N, A = len(X), self.n, self.N, self.A
        self.n_eval += P
        J = inst.c_x[self.ar_n, X].sum(axis=1)
        lat = inst.d_proc[self.ar_n, X].sum(axis=1)
        rows = np.repeat(np.arange(P), n)
        flat = rows * N + X.ravel()
        cpu_load = np.bincount(flat, weights=np.tile(inst.cpu, P), minlength=P * N).reshape(P, N)
        mem_load = np.bincount(flat, weights=np.tile(inst.mem, P), minlength=P * N).reshape(P, N)
        used = cpu_load > 0
        J += used @ inst.c_y
        if inst.c_h is not None:                                    # headroom-preserving cost
            excess = np.maximum(0.0, (inst.used_cpu[None, :] + cpu_load) / inst.sub.cpu_cap[None, :] - inst.theta_h)
            excess -= inst.excess0[None, :]
            J += np.maximum(0.0, excess) @ inst.c_h_vec
        sub = inst.sub
        v_node = (np.maximum(0.0, cpu_load - inst.res_cpu[None, :]) / sub.cpu_cap[None, :]).sum(axis=1) \
            + (np.maximum(0.0, mem_load - inst.res_mem[None, :]) / sub.mem_cap[None, :]).sum(axis=1)
        nodes = np.empty((P, n + 2), dtype=np.int64)
        nodes[:, 0] = self.src
        nodes[:, 1:-1] = X
        nodes[:, -1] = self.dst
        disc = np.zeros(P)
        arc_rows, arc_ids = [], []
        for p in range(P):
            row = nodes[p]
            Fp = F[p]
            for e in range(n + 1):
                u, v = int(row[e]), int(row[e + 1])
                if u == v:
                    continue
                arr, plat, pcost = self._path(u, v, int(Fp[e]))
                if arr is None:
                    disc[p] += 1
                    continue
                lat[p] += plat
                J[p] += pcost
                if len(arr):
                    arc_ids.append(arr)
                    arc_rows.append(np.full(len(arr), p))
        if arc_ids:
            ids = np.concatenate(arc_ids)
            rws = np.concatenate(arc_rows)
            load = np.bincount(rws * A + ids, minlength=P * A).reshape(P, A) * self.bw
            v_link = (np.maximum(0.0, load - inst.res_bw[None, :]) / sub.arc_cap[None, :]).sum(axis=1)
        else:
            v_link = np.zeros(P)
        v_sla = np.maximum(0.0, lat - inst.req.sla_ms) / inst.req.sla_ms
        if cfg.prune:
            v_pol = 0.0
        else:
            v_pol = (~inst.allowed[self.ar_n, X]).sum(axis=1)
        V = cfg.w_node * v_node + cfg.w_link * v_link + cfg.w_sla * v_sla + cfg.w_pol * v_pol + 10.0 * disc
        V = np.where(V <= 1e-12, 0.0, V)
        if cfg.penalty == "static":
            phi = J + cfg.omega * V
        else:
            phi = np.where(V > 0, self.J_ub + V, J)
        return phi, J, V, lat

    # ------------------------------------------------------------------ seeds
    def _random_pop(self, P):
        X = np.empty((P, self.n), dtype=np.int64)
        for i, d in enumerate(self.dom):
            X[:, i] = d[self.rng.integers(len(d), size=P)]
        F = self.rng.integers(0, self.cfg.k_paths, (P, self.n + 1))
        return X, F

    def _greedy_seed(self):
        inst = self.inst
        dist = self.router.dist()
        res_cpu, res_mem = inst.res_cpu.copy(), inst.res_mem.copy()
        prev, X = self.src, []
        for i in range(self.n):
            d = self.dom[i]
            ok = d[(res_cpu[d] >= inst.cpu[i] - 1e-9) & (res_mem[d] >= inst.mem[i] - 1e-9)]
            if len(ok) == 0:
                return None
            score = dist[prev, ok] + inst.d_proc[i, ok]
            if i == self.n - 1:
                score = score + dist[ok, self.dst]
            j = int(ok[np.argmin(score)])
            res_cpu[j] -= inst.cpu[i]
            res_mem[j] -= inst.mem[i]
            X.append(j)
            prev = j
        return np.array(X, dtype=np.int64)

    def _consolidation_seed(self):
        inst = self.inst
        common = set(self.dom[0].tolist())
        for d in self.dom[1:]:
            common &= set(d.tolist())
        if not common:
            return None
        cand = np.array(sorted(common))
        ok = cand[(inst.res_cpu[cand] >= inst.cpu.sum() - 1e-9) & (inst.res_mem[cand] >= inst.mem.sum() - 1e-9)]
        if len(ok) == 0:
            return None
        dist = self.router.dist()
        score = dist[self.src, ok] + dist[ok, self.dst] + inst.d_proc[:, ok].sum(axis=0) + 1e3 * (~inst.active[ok])
        return np.full(self.n, int(ok[np.argmin(score)]), dtype=np.int64)

    def _archive_key(self):
        r = self.inst.req
        return (r.template, r.ingress, r.egress)

    def _seeds(self):
        out = []
        if self.archive is not None:
            for X, F in self.archive.get(self._archive_key(), []):
                if all(inst_ok for inst_ok in (X[i] in self._domset[i] for i in range(self.n))):
                    out.append((np.array(X), np.array(F)))
        for X in (self._greedy_seed(), self._consolidation_seed()):
            if X is not None:
                out.append((X, np.zeros(self.n + 1, dtype=np.int64)))
        return out

    # ------------------------------------------------------------------ operators (vectorised)
    def _select(self, fit, m):
        idx = self.rng.integers(0, len(fit), (m, self.cfg.tour))
        return idx[np.arange(m), np.argmin(fit[idx], axis=1)]

    def _offspring(self, X, F, fit, m):
        cfg = self.cfg
        a, b = self._select(fit, m), self._select(fit, m)
        Xc, Fc = X[a].copy(), F[a].copy()
        do = self.rng.random(m) < cfg.pc
        if self.n > 1:
            cut = self.rng.integers(1, self.n + 1, m)
        else:
            cut = np.ones(m, dtype=np.int64)
        mx = (np.arange(self.n)[None, :] >= cut[:, None]) & do[:, None]
        mf = (np.arange(self.n + 1)[None, :] >= cut[:, None]) & do[:, None]
        Xc[mx] = X[b][mx]
        Fc[mf] = F[b][mf]
        mut = self.rng.random((m, self.n)) < cfg.pm
        for i in range(self.n):
            k = int(mut[:, i].sum())
            if k:
                d = self.dom[i]
                Xc[mut[:, i], i] = d[self.rng.integers(len(d), size=k)]
        mutf = self.rng.random((m, self.n + 1)) < cfg.pm
        k = int(mutf.sum())
        if k:
            Fc[mutf] = self.rng.integers(0, cfg.k_paths, k)
        return Xc, Fc

    # ------------------------------------------------------------------ local search (batched)
    def _local_search(self, X, F, best, budget):
        start = self.n_eval
        improved = True
        while improved and self.n_eval - start < budget:
            improved = False
            for i in range(self.n):
                d = self.dom[i]
                d = d[d != X[i]]
                if len(d) == 0 or self.n_eval - start >= budget:
                    continue
                d = d[: max(1, budget - (self.n_eval - start))]
                Xn = np.repeat(X[None, :], len(d), axis=0)
                Xn[:, i] = d
                Fn = np.repeat(F[None, :], len(d), axis=0)
                Fn[:, i] = 0
                Fn[:, i + 1] = 0
                phi, J, V, lat = self.evaluate(Xn, Fn)
                k = int(np.argmin(phi))
                if phi[k] < best[0] - 1e-12:
                    X, F, best, improved = Xn[k].copy(), Fn[k].copy(), (phi[k], J[k], V[k], lat[k]), True
            for e in range(self.n + 1):
                if self.n_eval - start >= budget:
                    break
                ks = np.array([k for k in range(self.cfg.k_paths) if k != F[e]])
                if len(ks) == 0:
                    continue
                Fn = np.repeat(F[None, :], len(ks), axis=0)
                Fn[:, e] = ks
                Xn = np.repeat(X[None, :], len(ks), axis=0)
                phi, J, V, lat = self.evaluate(Xn, Fn)
                k = int(np.argmin(phi))
                if phi[k] < best[0] - 1e-12:
                    F, best, improved = Fn[k].copy(), (phi[k], J[k], V[k], lat[k]), True
        return X, F, best

    # ------------------------------------------------------------------ main loop
    def run(self, history: bool = False):
        cfg = self.cfg
        if any(len(d) == 0 for d in self.dom):
            return dict(servers=None, paths=None, status="infeasible_domain", n_eval=0, best=None, hist=[], gens=0)
        self._domset = [set(d.tolist()) for d in self.dom]
        budget = cfg.pop * cfg.gens
        ls_budget = int(cfg.ls_share * budget) if cfg.local_search else 0
        gen_budget = budget - ls_budget
        X, F = self._random_pop(cfg.pop)
        if cfg.warm:
            seeds = self._seeds()[: max(1, cfg.pop // 10)]
            for k, (sx, sf) in enumerate(seeds):
                X[k], F[k] = sx, sf
        phi, J, V, lat = self.evaluate(X, F)
        hist = []
        gen = 0
        m = cfg.pop - cfg.elite
        while self.n_eval + m <= gen_budget:
            gen += 1
            order = np.argsort(phi, kind="stable")
            el = order[: cfg.elite]
            Xc, Fc = self._offspring(X, F, phi, m)
            p2, J2, V2, l2 = self.evaluate(Xc, Fc)
            X = np.concatenate([X[el], Xc])
            F = np.concatenate([F[el], Fc])
            phi = np.concatenate([phi[el], p2])
            J = np.concatenate([J[el], J2])
            V = np.concatenate([V[el], V2])
            lat = np.concatenate([lat[el], l2])
            if history:
                b = int(np.argmin(phi))
                hist.append((self.n_eval, float(phi[b]), float(J[b]), float(V[b])))
        b = int(np.argmin(phi))
        Xb, Fb, best = X[b].copy(), F[b].copy(), (phi[b], J[b], V[b], lat[b])
        if cfg.local_search and ls_budget > 0:
            Xb, Fb, best = self._local_search(Xb, Fb, best, ls_budget)
            if history:
                hist.append((self.n_eval, float(best[0]), float(best[1]), float(best[2])))
        feasible = best[2] <= 1e-12
        servers = [int(j) for j in Xb]
        nodes = [self.src] + servers + [self.dst]
        paths = []
        for e in range(self.n + 1):
            u, v = nodes[e], nodes[e + 1]
            if u == v:
                paths.append([])
            else:
                arr = self._path(u, v, int(Fb[e]))[0]
                paths.append([] if arr is None else [int(a) for a in arr])
        if feasible and self.archive is not None:
            lst = self.archive.setdefault(self._archive_key(), [])
            lst.insert(0, (Xb.copy(), Fb.copy()))
            del lst[3:]
        return dict(servers=servers, paths=paths, status="feasible" if feasible else "infeasible",
                    n_eval=self.n_eval, best=dict(phi=float(best[0]), J=float(best[1]), V=float(best[2]),
                                                  lat=float(best[3])), hist=hist, gens=gen)


def solve(inst: Instance, router: Router, cfg: GAConfig, archive: dict | None = None, history: bool = False):
    t0 = time.perf_counter()
    eng = GAEngine(inst, router, cfg, archive)
    res = eng.run(history=history)
    res["t_solve"] = time.perf_counter() - t0
    return res
