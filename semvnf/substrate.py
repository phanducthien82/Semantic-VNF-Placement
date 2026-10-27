"""Seed-dependent substrate attributes (Table 7) and the dynamic resource state.

Static attributes are drawn once per seed; the dynamic state (residual capacities,
tenants, operational state) evolves during the online simulation.  All draws use
``numpy.random.default_rng([seed, stream])`` so that every algorithm sees an identical
substrate for a given seed (paired experimental design, Section 4.2).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .catalog import CAPABILITIES, EDGE_SLOWDOWN
from .topology import Topology

CI_LEVELS = np.array([60.0, 150.0, 280.0, 420.0, 650.0])   # gCO2eq/kWh, representative grid mixes
CAP_PROB = {  # P(server has capability) for (core, edge) sites
    "GPU": (0.6, 0.2), "SmartNIC": (0.6, 0.4), "CryptoAccel": (0.7, 0.3)}
PKT_BITS = 1500 * 8


@dataclass
class Substrate:
    topo: Topology
    seed: int
    # ---- static node attributes
    cpu_cap: np.ndarray        # cores
    mem_cap: np.ndarray        # GB
    p_idle: np.ndarray         # W
    p_max: np.ndarray          # W
    trust: np.ndarray          # trust level of the server's security domain (1..3)
    caps: np.ndarray           # (n, |CAPABILITIES|) bool
    zone: np.ndarray           # zone id = 2*region + is_core
    zone_access_ms: np.ndarray # access latency of each zone
    region_ci: np.ndarray      # carbon intensity per region [gCO2eq/kWh]
    proc_factor: np.ndarray    # processing slow-down factor (1.0 core, 1.25 edge)
    # ---- static arc attributes
    arcs: np.ndarray           # (A, 2)
    arc_cap: np.ndarray        # Mbps
    arc_prop: np.ndarray       # ms
    arc_tx: np.ndarray         # ms (one 1500-byte packet)
    # ---- dynamic state
    cpu_used: np.ndarray = None
    mem_used: np.ndarray = None
    arc_used: np.ndarray = None
    node_up: np.ndarray = None
    arc_up: np.ndarray = None
    n_vnfs: np.ndarray = None
    mc_vnfs: np.ndarray = None
    exclusive: np.ndarray = None
    tenants: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # ------------------------------------------------------------------ derived
    @property
    def n(self) -> int:
        return self.topo.n

    @property
    def core(self) -> np.ndarray:
        return self.topo.core

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    @property
    def server_ci(self) -> np.ndarray:
        return self.region_ci[self.topo.region]

    @property
    def server_access_ms(self) -> np.ndarray:
        return self.zone_access_ms[self.zone]

    def reset_state(self) -> None:
        n, a = self.n, self.n_arcs
        self.cpu_used = np.zeros(n)
        self.mem_used = np.zeros(n)
        self.arc_used = np.zeros(a)
        self.node_up = np.ones(n, dtype=bool)
        self.arc_up = np.ones(a, dtype=bool)
        self.n_vnfs = np.zeros(n, dtype=int)
        self.mc_vnfs = np.zeros(n, dtype=int)
        self.exclusive = np.zeros(n, dtype=bool)
        self.tenants = defaultdict(lambda: defaultdict(int))

    def res_cpu(self) -> np.ndarray:
        return self.cpu_cap - self.cpu_used

    def res_mem(self) -> np.ndarray:
        return self.mem_cap - self.mem_used

    def res_bw(self) -> np.ndarray:
        return self.arc_cap - self.arc_used

    def utilization(self) -> np.ndarray:
        return self.cpu_used / self.cpu_cap

    def active(self) -> np.ndarray:
        return self.n_vnfs > 0

    def tenant_count(self) -> np.ndarray:
        return np.array([len(self.tenants[j]) if j in self.tenants else 0 for j in range(self.n)])

    def usable_arcs(self) -> np.ndarray:
        """Arcs whose link and both endpoints are operational."""
        up = self.arc_up & self.node_up[self.arcs[:, 0]] & self.node_up[self.arcs[:, 1]]
        return up

    # ------------------------------------------------------------------ embedding bookkeeping
    def allocate(self, emb) -> None:
        for (j, cpu, mem, mc, iso) in emb.node_alloc:
            self.cpu_used[j] += cpu
            self.mem_used[j] += mem
            self.n_vnfs[j] += 1
            self.mc_vnfs[j] += int(mc)
            if iso:
                self.exclusive[j] = True
            self.tenants[j][emb.sfc_id] += 1
        for a, bw in emb.arc_alloc:
            self.arc_used[a] += bw

    def release(self, emb) -> None:
        for (j, cpu, mem, mc, iso) in emb.node_alloc:
            self.cpu_used[j] -= cpu
            self.mem_used[j] -= mem
            self.n_vnfs[j] -= 1
            self.mc_vnfs[j] -= int(mc)
            t = self.tenants[j]
            t[emb.sfc_id] -= 1
            if t[emb.sfc_id] <= 0:
                del t[emb.sfc_id]
            if iso:
                self.exclusive[j] = False
        for a, bw in emb.arc_alloc:
            self.arc_used[a] -= bw
        # guard against floating point drift
        self.cpu_used[np.abs(self.cpu_used) < 1e-9] = 0.0
        self.mem_used[np.abs(self.mem_used) < 1e-9] = 0.0
        self.arc_used[np.abs(self.arc_used) < 1e-6] = 0.0


def make_substrate(topo: Topology, seed: int) -> Substrate:
    """Draw the static attributes of all servers and links for ``seed`` (documented in Table 7)."""
    rng = np.random.default_rng([seed, 11])
    n, core = topo.n, topo.core
    cpu = np.where(core, rng.integers(24, 33, n), rng.integers(8, 17, n)).astype(float)
    mem = 4.0 * cpu
    p_idle = np.where(core, rng.uniform(150, 200, n), rng.uniform(60, 90, n))
    p_max = np.where(core, rng.uniform(350, 450, n), rng.uniform(150, 220, n))
    trust = np.where(core, rng.choice([2, 3], n, p=[0.3, 0.7]), rng.choice([1, 2], n, p=[0.4, 0.6]))
    caps = np.zeros((n, len(CAPABILITIES)), dtype=bool)
    for k, c in enumerate(CAPABILITIES):
        pc, pe = CAP_PROB[c]
        caps[:, k] = rng.random(n) < np.where(core, pc, pe)
    n_reg = topo.n_regions
    zone = 2 * topo.region + core.astype(int)
    zone_access = np.empty(2 * n_reg)
    zone_access[0::2] = rng.uniform(0.5, 2.5, n_reg)   # edge zones
    zone_access[1::2] = rng.uniform(3.0, 8.0, n_reg)   # core zones
    reps = int(np.ceil(n_reg / len(CI_LEVELS)))
    region_ci = rng.permutation(np.tile(CI_LEVELS, reps))[:n_reg]
    proc_factor = np.where(core, 1.0, EDGE_SLOWDOWN)
    arcs = topo.arcs()
    link_cap = rng.integers(1, 11, topo.m) * 1000.0           # U{1..10} Gbps, full duplex
    arc_cap = np.repeat(link_cap, 2)
    arc_prop = topo.arc_prop_ms()
    arc_tx = PKT_BITS / (arc_cap * 1e6) * 1e3
    sub = Substrate(topo=topo, seed=seed, cpu_cap=cpu, mem_cap=mem, p_idle=p_idle, p_max=p_max,
                    trust=trust.astype(int), caps=caps, zone=zone, zone_access_ms=zone_access,
                    region_ci=region_ci, proc_factor=proc_factor, arcs=arcs, arc_cap=arc_cap,
                    arc_prop=arc_prop, arc_tx=arc_tx)
    sub.reset_state()
    return sub
