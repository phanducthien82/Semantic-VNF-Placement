"""Substrate topologies.

Three families are used in the paper (Section 4.1, Table 7):

* ``euro28``    -- SNDlib ``nobel-eu`` (28 nodes, 41 links), real geography;
* ``germany50`` -- SNDlib ``germany50`` (50 nodes, 88 links), real geography;
* ``ba-<N>``    -- Barabasi-Albert graphs (preferential attachment, m=2) used only
                   for the scalability study (N = 50 ... 1000).

Link propagation latency is derived from great-circle distances for SNDlib graphs
(fibre route factor 1.5, 200 km/ms in fibre) and drawn from U(1, 5) ms for BA graphs.
Every undirected link is full duplex: it becomes two directed arcs with independent
capacity.  Arc ``2k`` is (u -> v) and arc ``2k+1`` is (v -> u) for edge ``k = (u, v)``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sndlib"
SNDLIB_FILES = {"euro28": "nobel-eu.txt", "germany50": "germany50.txt"}
SNDLIB_SIZES = {"euro28": (28, 41), "germany50": (50, 88)}

FIBRE_KM_PER_MS = 200.0     # signal speed in fibre (2e5 km/s)
ROUTE_FACTOR = 1.5          # fibre route length / great-circle distance
EARTH_RADIUS_KM = 6371.0


@dataclass
class Topology:
    """Structure and geography of a substrate network (attributes are added in substrate.py)."""

    name: str
    n: int
    edges: np.ndarray                 # (m, 2) undirected, u < v
    prop_ms: np.ndarray               # (m,) one-way propagation latency per link [ms]
    names: list[str]
    pos: np.ndarray | None = None     # (n, 2) lon/lat (SNDlib only)
    core: np.ndarray | None = None    # (n,) bool: core (True) or edge (False) site
    region: np.ndarray | None = None  # (n,) int: geographical region id (for zones / carbon)
    source: str = "sndlib"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ arcs
    @property
    def m(self) -> int:
        return int(len(self.edges))

    @property
    def n_regions(self) -> int:
        return int(self.region.max()) + 1

    def arcs(self) -> np.ndarray:
        """(2m, 2) directed arcs; arc 2k = (u, v), arc 2k+1 = (v, u)."""
        a = np.empty((2 * self.m, 2), dtype=np.int64)
        a[0::2] = self.edges
        a[1::2] = self.edges[:, ::-1]
        return a

    def arc_prop_ms(self) -> np.ndarray:
        return np.repeat(self.prop_ms, 2)

    def graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for k, (u, v) in enumerate(self.edges):
            g.add_edge(int(u), int(v), k=k, lat=float(self.prop_ms[k]))
        return g


# ---------------------------------------------------------------------- helpers
def _haversine_km(p: np.ndarray, q: np.ndarray) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _canon(edges) -> np.ndarray:
    e = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if u != v})
    return np.array(e, dtype=np.int64).reshape(-1, 2)


def _core_by_degree(n: int, edges: np.ndarray, frac: float) -> np.ndarray:
    deg = np.bincount(edges.ravel(), minlength=n)
    k = max(1, int(round(frac * n)))
    order = sorted(range(n), key=lambda j: (-deg[j], j))  # deterministic tie-break
    core = np.zeros(n, dtype=bool)
    core[order[:k]] = True
    return core


def _kmeans_regions(pos: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    """Deterministic k-means (farthest-point initialisation) on lon/lat."""
    x = np.asarray(pos, dtype=float)
    centres = [x[np.argmin(x[:, 0])]]                  # west-most node first
    while len(centres) < k:
        d = np.min([np.sum((x - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(x[int(np.argmax(d))])
    c = np.array(centres)
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        lab_new = np.argmin(((x[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        if np.array_equal(lab_new, lab) and _ > 0:
            break
        lab = lab_new
        for r in range(k):
            if np.any(lab == r):
                c[r] = x[lab == r].mean(axis=0)
    # relabel regions by west-to-east order of centroids for readability
    order = np.argsort(c[:, 0])
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[lab]


def _hop_regions(g: nx.Graph, n_regions: int) -> np.ndarray:
    """Regions for non-geographic graphs: hop-distance Voronoi cells around the highest-degree hubs."""
    hubs = sorted(g.nodes, key=lambda j: (-g.degree[j], j))[:n_regions]
    dist = {h: nx.single_source_shortest_path_length(g, h) for h in hubs}
    reg = np.zeros(g.number_of_nodes(), dtype=int)
    for j in g.nodes:
        reg[j] = min(range(len(hubs)), key=lambda r: (dist[hubs[r]].get(j, 10 ** 9), r))
    return reg


# ---------------------------------------------------------------------- loaders
def load_sndlib(name: str) -> Topology:
    """Parse an SNDlib native network file (``data/sndlib``)."""
    key = name.lower()
    path = DATA_DIR / SNDLIB_FILES[key]
    text = path.read_text(encoding="utf-8", errors="replace")
    nodes_blk = re.search(r"NODES\s*\((.*?)\n\)", text, re.S)
    links_blk = re.search(r"LINKS\s*\((.*?)\n\)", text, re.S)
    names, pos, raw = [], [], []
    for line in nodes_blk.group(1).splitlines():
        mm = re.match(r"\s*(\S+)\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", line)
        if mm:
            names.append(mm.group(1))
            pos.append((float(mm.group(2)), float(mm.group(3))))
    idx = {nm: i for i, nm in enumerate(names)}
    for line in links_blk.group(1).splitlines():
        mm = re.match(r"\s*(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)", line)
        if mm:
            raw.append((idx[mm.group(2)], idx[mm.group(3)]))
    edges = _canon(raw)
    pos = np.array(pos)
    n = len(names)
    if (n, len(edges)) != SNDLIB_SIZES[key]:
        raise ValueError(f"{path}: expected {SNDLIB_SIZES[key]}, parsed {(n, len(edges))}")
    km = np.array([_haversine_km(pos[u], pos[v]) for u, v in edges])
    prop = km * ROUTE_FACTOR / FIBRE_KM_PER_MS
    core = _core_by_degree(n, edges, frac=0.30)
    n_reg = {"euro28": 5, "germany50": 5}[key]
    region = _kmeans_regions(pos, n_reg)
    return Topology(name=key, n=n, edges=edges, prop_ms=prop, names=names, pos=pos, core=core,
                    region=region, source="sndlib",
                    meta={"file": path.name, "km_mean": float(km.mean()), "km_max": float(km.max())})


def barabasi_albert(n: int, seed: int, m_attach: int = 2, lat_range=(1.0, 5.0)) -> Topology:
    """BA graph used in the scalability study (Section 4.6)."""
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(n, m_attach, seed=int(rng.integers(2 ** 31 - 1)))
    edges = _canon(g.edges)
    prop = rng.uniform(lat_range[0], lat_range[1], size=len(edges))
    core = _core_by_degree(n, edges, frac=0.20)
    n_reg = max(3, int(round(n / 12)))
    region = _hop_regions(g, n_reg)
    return Topology(name=f"ba-{n}", n=n, edges=edges, prop_ms=prop, names=[f"n{j}" for j in range(n)],
                    core=core, region=region, source="synthetic-ba",
                    meta={"m_attach": m_attach, "lat_range": list(lat_range), "graph_seed": seed})


def load_topology(name: str, seed: int = 0) -> Topology:
    """``euro28`` | ``germany50`` | ``ba-<N>``."""
    key = name.lower()
    if key in SNDLIB_FILES:
        return load_sndlib(key)
    if key.startswith("ba-"):
        return barabasi_albert(int(key.split("-")[1]), seed=seed)
    raise KeyError(name)
