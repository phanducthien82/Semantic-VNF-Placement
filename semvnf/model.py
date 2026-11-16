"""Single-request placement problem (Section 3.4), objective coefficients, solution
evaluation, latency decomposition and the algorithm-independent admission validator.

Notation follows the revised manuscript:
  x_ij  VNF i on server j,   y_j  activation of an idle server j,
  f^e_a virtual link e (e = 0..n, including ingress and egress links) on arc a.

Objective (all terms dimensionless, Eq. (24) of the revision)::

  J = wL*L/L_ref + wE*dP/P_ref + wC*dG/G_ref + wU*U/n

with L the end-to-end latency [ms], dP the incremental power [W], dG the incremental
carbon emission rate [gCO2eq/h] and U the load-aware utilisation cost.  Every term is
linear in (x, y, f) because all denominators are constants of the decision epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalog import VNFSpec
from .substrate import Substrate

DEFAULT_WEIGHTS = dict(wL=0.4, wE=0.3, wC=0.2, wU=0.1, wH=2.0, wS=1.0)
HEADROOM_THETA = 0.8         # utilisation safety margin of the mission-critical policy (P6)
PI_TX_W_PER_MBPS = 0.01      # transport power per Mbps per traversed hop (10 nJ/bit)


@dataclass
class Request:
    sfc_id: int
    template: str
    ingress: int
    egress: int
    vnfs: list            # list[VNFSpec]
    bw: float             # Mbps carried by every virtual link
    sla_ms: float
    arrival: float = 0.0
    holding: float = 0.0

    @property
    def n(self) -> int:
        return len(self.vnfs)


@dataclass
class Embedding:
    sfc_id: int
    servers: list
    paths: list            # n+1 arc lists
    node_alloc: list       # (j, cpu, mem, mission_critical, isolated)
    arc_alloc: list        # (arc, bw)
    latency: float = 0.0
    lat_prop: float = 0.0
    lat_tx: float = 0.0
    lat_proc: float = 0.0
    hops: int = 0
    request: Request | None = None
    expires: float = 0.0


@dataclass
class Instance:
    """Snapshot of everything an algorithm needs for one decision epoch."""
    req: Request
    sub: Substrate
    allowed: np.ndarray       # (n, N) semantic candidates  (True = not in I_sem)
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    pi_tx: float = PI_TX_W_PER_MBPS
    kv: object = None         # knowledge view (for policy-scarcity pricing)

    def __post_init__(self):
        s, r = self.sub, self.req
        self.n, self.N = r.n, s.n
        self.cpu = np.array([v.cpu for v in r.vnfs], dtype=float)
        self.mem = np.array([v.mem for v in r.vnfs], dtype=float)
        self.res_cpu = s.res_cpu().copy()
        self.res_mem = s.res_mem().copy()
        self.res_bw = s.res_bw().copy()
        self.node_up = s.node_up.copy()
        self.usable = s.usable_arcs()
        self.arc_lat = s.arc_prop + s.arc_tx
        self.active = s.active().copy()
        self.d_proc = np.outer([v.proc for v in r.vnfs], s.proc_factor)          # (n, N) ms
        self.cap_ok = ((self.cpu[:, None] <= self.res_cpu[None, :] + 1e-9)
                       & (self.mem[:, None] <= self.res_mem[None, :] + 1e-9)
                       & self.node_up[None, :])
        self.cand = self.allowed & self.cap_ok                                   # domains C_i
        self.ci = s.server_ci
        self.ci_net = float(np.mean(s.region_ci))
        w = self.weights
        self.L_ref = r.sla_ms
        self.P_ref = float(np.mean(s.p_idle))
        self.G_ref = self.P_ref / 1000.0 * self.ci_net
        dyn = (s.p_max - s.p_idle)[None, :] * self.cpu[:, None] / s.cpu_cap[None, :]   # (n, N) W
        self.dP_x = dyn
        safe_res = np.maximum(self.res_cpu, 1e-6)
        self.c_x = (w["wL"] * self.d_proc / self.L_ref
                    + w["wE"] * dyn / self.P_ref
                    + w["wC"] * dyn / 1000.0 * self.ci[None, :] / self.G_ref
                    + w["wU"] * (self.cpu[:, None] / safe_res[None, :]) / self.n)
        idle_new = np.where(self.active, 0.0, s.p_idle)                            # W if j is switched on
        self.dP_y = idle_new
        self.c_y = w["wE"] * idle_new / self.P_ref + w["wC"] * idle_new / 1000.0 * self.ci / self.G_ref
        p_tx = self.pi_tx * r.bw                                                   # W per traversed hop
        self.c_f = (w["wL"] * self.arc_lat / self.L_ref + w["wE"] * p_tx / self.P_ref
                    + w["wC"] * p_tx / 1000.0 * self.ci_net / self.G_ref)
        self.endpoints_src = r.ingress
        self.endpoints_dst = r.egress
        # headroom-preserving cost: w_H * sum_j z_j, z_j = incremental utilisation above theta
        self.theta_h = HEADROOM_THETA
        self.used_cpu = s.cpu_used.copy()
        self.excess0 = np.maximum(0.0, s.cpu_used / s.cpu_cap - self.theta_h)
        wH = float(w.get("wH", 0.0))
        self.c_h = wH if wH > 0 else None
        self.c_h_vec = np.full(self.N, wH)
        # policy-scarcity pricing (Section 3.4.3)
        wS = float(w.get("wS", 0.0))
        self.c_psp = None
        if wS > 0:
            from .pricing import psp_coefficients
            from .rules import view_from_substrate
            kv = self.kv if self.kv is not None else view_from_substrate(s)
            self.c_psp = psp_coefficients(r.vnfs, s, kv, self.res_cpu, wS)
            self.c_x = self.c_x + self.c_psp

    # ------------------------------------------------------------------ objective of a decoded solution
    def evaluate(self, servers, paths) -> dict:
        """Objective, latency decomposition and constraint violations of (servers, paths)."""
        r, s = self.req, self.sub
        servers = [int(j) for j in servers]
        lat_proc = float(sum(self.d_proc[i, j] for i, j in enumerate(servers)))
        arcs = [a for p in paths for a in p]
        lat_prop = float(sum(s.arc_prop[a] for a in arcs))
        lat_tx = float(sum(s.arc_tx[a] for a in arcs))
        lat = lat_proc + lat_prop + lat_tx
        used = sorted(set(servers))
        J_x = float(sum(self.c_x[i, j] for i, j in enumerate(servers)))
        J_y = float(sum(self.c_y[j] for j in used))
        J_f = float(sum(self.c_f[a] for a in arcs))
        J_h = 0.0
        if self.c_h is not None:
            add = {}
            for i, j in enumerate(servers):
                add[j] = add.get(j, 0.0) + self.cpu[i]
            for j, l in add.items():
                ex = max(0.0, (self.used_cpu[j] + l) / s.cpu_cap[j] - self.theta_h) - self.excess0[j]
                J_h += self.c_h_vec[j] * max(0.0, ex)
        dP = float(sum(self.dP_x[i, j] for i, j in enumerate(servers)) + sum(self.dP_y[j] for j in used)
                   + self.pi_tx * r.bw * len(arcs))
        # violations
        cpu_load = {}
        mem_load = {}
        for i, j in enumerate(servers):
            cpu_load[j] = cpu_load.get(j, 0.0) + self.cpu[i]
            mem_load[j] = mem_load.get(j, 0.0) + self.mem[i]
        v_node = sum(max(0.0, cpu_load[j] - self.res_cpu[j]) / s.cpu_cap[j]
                     + max(0.0, mem_load[j] - self.res_mem[j]) / s.mem_cap[j] for j in cpu_load)
        arc_load = {}
        for a in arcs:
            arc_load[a] = arc_load.get(a, 0.0) + r.bw
        v_link = sum(max(0.0, arc_load[a] - self.res_bw[a]) / s.arc_cap[a] for a in arc_load)
        v_sla = max(0.0, lat - r.sla_ms) / r.sla_ms
        v_pol = int(sum(0 if self.allowed[i, j] else 1 for i, j in enumerate(servers)))
        v_down = int(sum(0 if self.node_up[j] else 1 for j in servers) + sum(0 if self.usable[a] else 1 for a in arcs))
        return dict(J=J_x + J_y + J_f + J_h, latency=lat, lat_proc=lat_proc, lat_prop=lat_prop, lat_tx=lat_tx,
                    dP=dP, hops=len(arcs), v_node=v_node, v_link=v_link, v_sla=v_sla, v_pol=v_pol,
                    v_down=v_down, n_servers=len(used))

    def path_endpoints(self, servers):
        nodes = [self.req.ingress] + [int(j) for j in servers] + [self.req.egress]
        return list(zip(nodes[:-1], nodes[1:]))

    # ------------------------------------------------------------------ objective upper bound
    def J_upper_bound(self) -> float:
        """Instance-wise upper bound on J over all (possibly infeasible) decodings (Proposition 3)."""
        n = self.n
        bx = float(np.max(self.c_x, axis=1).sum())
        by = float(np.sort(self.c_y)[::-1][:n].sum())
        usable = self.c_f[self.usable]
        bf = float((n + 1) * usable.sum())          # a simple path uses each arc at most once
        bh = float(self.c_h_vec.max() * self.cpu.sum() / self.sub.cpu_cap.min()) if self.c_h is not None else 0.0
        return bx + by + bf + bh + 1.0

    def to_embedding(self, servers, paths) -> Embedding:
        r = self.req
        ev = self.evaluate(servers, paths)
        node_alloc = [(int(j), float(v.cpu), float(v.mem), "MissionCriticalVNF" in v.classes,
                       "IsolatedVNF" in v.classes) for v, j in zip(r.vnfs, servers)]
        arc_alloc = [(int(a), float(r.bw)) for p in paths for a in p]
        return Embedding(sfc_id=r.sfc_id, servers=[int(j) for j in servers], paths=[list(p) for p in paths],
                         node_alloc=node_alloc, arc_alloc=arc_alloc, latency=ev["latency"],
                         lat_prop=ev["lat_prop"], lat_tx=ev["lat_tx"], lat_proc=ev["lat_proc"],
                         hops=ev["hops"], request=r)


def validate(inst: Instance, servers, paths, truth_allowed: np.ndarray | None = None) -> tuple[bool, str]:
    """Algorithm-independent admission check (capacity, bandwidth, SLA, path continuity, policy).

    ``truth_allowed`` is the ground-truth policy mask; it equals ``inst.allowed`` except in the
    imperfect-knowledge experiment (Section 4.9)."""
    r = inst.req
    if servers is None or paths is None or len(servers) != r.n or len(paths) != r.n + 1:
        return False, "no_solution"
    arcs_ = inst.sub.arcs
    for (u, v), p in zip(inst.path_endpoints(servers), paths):
        cur = u
        for a in p:
            if int(arcs_[a, 0]) != cur:
                return False, "path_broken"
            cur = int(arcs_[a, 1])
        if cur != v:
            return False, "path_broken"
    ev = inst.evaluate(servers, paths)
    if ev["v_down"] > 0:
        return False, "down_element"
    pol = truth_allowed if truth_allowed is not None else inst.allowed
    if any(not pol[i, j] for i, j in enumerate(servers)):
        return False, "policy"
    if ev["v_node"] > 1e-9:
        return False, "node_capacity"
    if ev["v_link"] > 1e-9:
        return False, "link_capacity"
    if ev["v_sla"] > 1e-9:
        return False, "sla"
    return True, "ok"


def remove_cycles(path_arcs, arcs, src, dst):
    """Extract a simple src->dst path from arcs carrying a binary unit flow.

    Under flow conservation, walking along unused arcs from ``src`` always reaches ``dst``;
    chronological loop erasure of that walk yields a simple path.  Arcs on detached cycles
    are never visited and are dropped (Proposition 2 of the revision)."""
    succ = {}
    for a in path_arcs:
        succ.setdefault(int(arcs[a, 0]), []).append(int(a))
    walk, cur = [], src
    while cur != dst:
        nxt = succ.get(cur)
        if not nxt or len(walk) > len(path_arcs):
            return None
        a = nxt.pop(0)
        walk.append(a)
        cur = int(arcs[a, 1])
    out_nodes, out_arcs, index = [src], [], {src: 0}
    for a in walk:
        v = int(arcs[a, 1])
        if v in index:                       # erase the loop closed at v
            k = index[v]
            for w in out_nodes[k + 1:]:
                del index[w]
            out_nodes, out_arcs = out_nodes[:k + 1], out_arcs[:k]
        else:
            out_arcs.append(a)
            out_nodes.append(v)
            index[v] = len(out_nodes) - 1
    return out_arcs
