"""Policy-scarcity pricing (PSP), Section 3.4.3.

The knowledge graph does not only prune infeasible (server, VNF) pairs; it also tells the
optimiser which server features are *scarce with respect to the policies of future
requests*.  For every server-side feature phi that appears in a policy rule (hardware
capability, trust level, low-latency zone, low-carbon zone) we compare

    delta_phi = share of the expected CPU demand that *requires* phi (service catalogue),
    sigma_phi = share of the residual CPU capacity offered by servers that *have* phi,

and price the scarcity s_phi = delta_phi / sigma_phi.  Placing VNF i on server j costs

    c^PSP_ij = w_S * (r_i,cpu / C_j) * sum_{phi in Feat(j), phi not in Req(i)} s_phi,

i.e. consuming a scarce feature without needing it is discouraged.  The term is linear in
x and its coefficients are recomputed at every decision epoch from the knowledge view.
"""
from __future__ import annotations

import numpy as np

from .catalog import CAPABILITIES, TEMPLATES, vnf_specs


def _vnf_requirements(v) -> set:
    req = set()
    if v.cap is not None and "HardwareAcceleratedVNF" in v.classes:
        req.add(f"cap:{v.cap}")
    for t in (2, 3):
        if v.trust >= t:
            req.add(f"trust>={t}")
    if "LowLatencyVNF" in v.classes and v.max_access_ms is not None:
        req.add(f"access<={v.max_access_ms:g}")
    if "GreenVNF" in v.classes and v.max_ci is not None:
        req.add(f"ci<={v.max_ci:g}")
    return req


def feature_names():
    names = [f"cap:{c}" for c in CAPABILITIES] + ["trust>=2", "trust>=3"]
    for t in TEMPLATES:
        for v in vnf_specs(t):
            names += [r for r in _vnf_requirements(v) if r.startswith(("access", "ci"))]
    return sorted(set(names), key=names.index)


FEATURES = feature_names()


def server_features(sub, kv) -> np.ndarray:
    """(N, |FEATURES|) bool matrix of the features offered by each server."""
    N = sub.n
    M = np.zeros((N, len(FEATURES)), dtype=bool)
    for f, name in enumerate(FEATURES):
        if name.startswith("cap:"):
            M[:, f] = kv.caps[:, CAPABILITIES.index(name[4:])]
        elif name.startswith("trust>="):
            M[:, f] = kv.trust >= int(name[7:])
        elif name.startswith("access<="):
            M[:, f] = kv.access_ms <= float(name[8:])
        elif name.startswith("ci<="):
            M[:, f] = kv.ci <= float(name[4:])
    return M


def demand_shares() -> np.ndarray:
    """Expected share of CPU demand requiring each feature (from the template mix)."""
    tot = 0.0
    d = np.zeros(len(FEATURES))
    for t in TEMPLATES:
        for v in vnf_specs(t):
            w = t.prob * v.cpu
            tot += w
            for r in _vnf_requirements(v):
                d[FEATURES.index(r)] += w
    return d / tot


DEMAND = demand_shares()


def psp_coefficients(vnfs, sub, kv, res_cpu, w_s: float) -> np.ndarray:
    """(n, N) additive objective coefficients of policy-scarcity pricing."""
    feats = server_features(sub, kv)
    res = np.maximum(res_cpu, 0.0) * kv.up
    tot = max(res.sum(), 1e-9)
    supply = (feats * res[:, None]).sum(axis=0) / tot
    scarcity = DEMAND / np.maximum(supply, 1e-3)
    out = np.zeros((len(vnfs), sub.n))
    for i, v in enumerate(vnfs):
        req = _vnf_requirements(v)
        mask = np.array([name not in req for name in FEATURES])
        price = feats[:, mask] @ scarcity[mask]                     # (N,)
        out[i] = w_s * v.cpu / sub.cpu_cap * price
    return out
