"""Routing utilities (Section 3.5.1).

Link weight.  Every operational arc a carries the weight w_a = l_a^prop + l_a^tx (ms), i.e.
the per-hop latency used in the objective; failed nodes/links are removed.

Candidate path set of the GA path genes.  For an ordered pair (u, v) the set P_K(u, v)
contains the latency-shortest path plus up to K-1 loop-free alternates (LFAs, RFC 5286
inequality 1) obtained from the all-pairs shortest-path tree:
  * first-hop alternates  u -> w -> SP(w, v)   for out-neighbours w with
                          d(w, v) < d(w, u) + d(u, v);
  * last-hop alternates   SP(u, w) -> w -> v   for in-neighbours w with
                          d(u, w) < d(u, v) + d(v, w);
sorted by total latency.  Every member is a simple path by construction, the set is
computed in O(deg(u) + deg(v)) path reconstructions from a precomputed shortest-path
tree, and therefore scales to the 1000-node graphs of the scalability study.  The exact
Yen K-shortest simple paths are available with ``method="yen"`` for comparison.
"""
from __future__ import annotations

import heapq
from itertools import islice

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Router:
    def __init__(self, n: int, arcs: np.ndarray, arc_lat: np.ndarray, usable: np.ndarray, k: int = 5,
                 method: str = "lfa"):
        self.n = n
        self.arcs = arcs
        self.arc_lat = arc_lat
        self.usable = usable.copy()
        self.k = k
        self.method = method
        self.out = [[] for _ in range(n)]
        self.inn = [[] for _ in range(n)]
        self.arc_id = {}
        for a in np.flatnonzero(self.usable):
            u, v = int(arcs[a, 0]), int(arcs[a, 1])
            self.out[u].append((int(a), v))
            self.inn[v].append((int(a), u))
            self.arc_id[(u, v)] = int(a)
        self._cache: dict = {}
        self._dist = None
        self._pred = None
        self._g = None

    # ------------------------------------------------------------------ all-pairs shortest paths
    def _apsp(self):
        if self._dist is None:
            idx = np.flatnonzero(self.usable)
            m = csr_matrix((self.arc_lat[idx], (self.arcs[idx, 0], self.arcs[idx, 1])), shape=(self.n, self.n))
            self._dist, self._pred = shortest_path(m, method="D", directed=True, return_predecessors=True)
        return self._dist, self._pred

    def dist(self) -> np.ndarray:
        return self._apsp()[0]

    def sp_arcs(self, u: int, v: int):
        """Arc list of the shortest path u->v (None if unreachable)."""
        if u == v:
            return []
        dist, pred = self._apsp()
        if not np.isfinite(dist[u, v]):
            return None
        nodes = [v]
        while nodes[-1] != u:
            nodes.append(int(pred[u, nodes[-1]]))
        nodes.reverse()
        return [self.arc_id[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]

    # ------------------------------------------------------------------ bandwidth-aware single path
    def shortest(self, s: int, t: int, bw: float = 0.0, res_bw: np.ndarray | None = None):
        """Latency-shortest path s->t over arcs with residual bandwidth >= bw (Dijkstra)."""
        if s == t:
            return [], 0.0
        if res_bw is not None and bw > 0:
            p = self.sp_arcs(s, t)                      # fast path: unconstrained SP is feasible
            if p is not None and all(res_bw[a] + 1e-9 >= bw for a in p):
                return p, float(self.dist()[s, t])
        distv = {s: 0.0}
        prev = {}
        pq = [(0.0, s)]
        done = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            if u == t:
                break
            done.add(u)
            for a, v in self.out[u]:
                if res_bw is not None and res_bw[a] + 1e-9 < bw:
                    continue
                nd = d + self.arc_lat[a]
                if nd < distv.get(v, np.inf):
                    distv[v] = nd
                    prev[v] = a
                    heapq.heappush(pq, (nd, v))
        if t not in prev:
            return None, np.inf
        path, v = [], t
        while v != s:
            a = prev[v]
            path.append(a)
            v = int(self.arcs[a, 0])
        path.reverse()
        return path, float(distv[t])

    # ------------------------------------------------------------------ candidate path sets
    def paths(self, u: int, v: int):
        """Candidate path set P_K(u, v) as a list of (arc list, latency), shortest first (cached)."""
        key = (u, v)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if u == v:
            res = [([], 0.0)]
        elif self.method == "yen":
            res = self._yen(u, v)
        else:
            res = self._lfa(u, v)
        self._cache[key] = res
        return res

    ksp = paths   # backwards-compatible name

    def _lfa(self, u, v):
        dist, _ = self._apsp()
        if not np.isfinite(dist[u, v]):
            return []
        prim = self.sp_arcs(u, v)
        out = [(prim, float(dist[u, v]))]
        if self.k <= 1:
            return out
        seen = {tuple(prim)}
        cands = []
        for a, w in self.out[u]:                                  # first-hop alternates
            if w == v:
                p = [a]
            elif np.isfinite(dist[w, v]) and dist[w, v] < dist[w, u] + dist[u, v] - 1e-12:
                p = [a] + self.sp_arcs(w, v)
            else:
                continue
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                cands.append((p, float(self.arc_lat[a] + (dist[w, v] if w != v else 0.0))))
        for a, w in self.inn[v]:                                  # last-hop alternates
            if w == u:
                p = [a]
            elif np.isfinite(dist[u, w]) and dist[u, w] < dist[u, v] + dist[v, w] - 1e-12:
                p = self.sp_arcs(u, w) + [a]
            else:
                continue
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                cands.append((p, float(dist[u, w] + self.arc_lat[a] if w != u else self.arc_lat[a])))
        cands.sort(key=lambda x: x[1])
        return out + cands[: self.k - 1]

    def _yen(self, u, v):
        if self._g is None:
            g = nx.DiGraph()
            g.add_nodes_from(range(self.n))
            for a in np.flatnonzero(self.usable):
                x, y = self.arcs[a]
                g.add_edge(int(x), int(y), a=int(a), w=float(self.arc_lat[a]))
            self._g = g
        g = self._g
        res = []
        try:
            for nodes in islice(nx.shortest_simple_paths(g, u, v, weight="w"), self.k):
                arcs = [g[a][b]["a"] for a, b in zip(nodes[:-1], nodes[1:])]
                res.append((arcs, float(sum(self.arc_lat[a] for a in arcs))))
        except nx.NetworkXNoPath:
            pass
        return res
