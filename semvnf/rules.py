"""Compiled (vectorised) evaluation of the seven policy types.

The RDF pipeline in kb.py is the reference semantics; this module evaluates exactly the
same DL-safe rules on numpy arrays extracted from the knowledge graph.  It is used inside
the online simulator, where the policies must be re-evaluated at every arrival.  The test
suite and experiment E0 check that both implementations return identical incompatibility
sets on every evaluated snapshot (Section 4.3).

Three-valued semantics: a rule fires only on validated (known) attributes.  Pairs whose
attributes are unknown are *not* inserted into I_sem (they stay feasible).  This is the default
mode ``unknown="permit"`` (sound: no feasible pair is lost).  The conservative mode
``unknown="deny"`` instead treats a pair as incompatible whenever an attribute that a rule needs
is unknown (complete with respect to missing or stale data, at the price of false positives);
it is applied when the placement snapshot is compiled, not by the Horn rules (Section 3.6, E7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import CAPABILITIES
from .substrate import Substrate

POLICIES = ("latency", "capability", "trust", "isolation", "opstate", "headroom", "carbon")
HEADROOM = 0.8


@dataclass
class KnowledgeView:
    """Policy-relevant attributes as known to the orchestrator (possibly imperfect)."""
    access_ms: np.ndarray          # (N,) access latency of the server's zone
    ci: np.ndarray                 # (N,) carbon intensity of the server's zone
    ci_known: np.ndarray           # (N,) bool: validated carbon observation
    trust: np.ndarray              # (N,) trust level of the server's domain
    caps: np.ndarray               # (N, C) bool
    inv_valid: np.ndarray          # (N,) bool: capability inventory validated (complete)
    up: np.ndarray                 # (N,) bool operational state
    util: np.ndarray               # (N,) utilisation
    tenants: np.ndarray            # (N,) active tenant count
    exclusive: np.ndarray          # (N,) bool
    state_known: np.ndarray        # (N,) bool: validated resource-state observation


def view_from_substrate(sub: Substrate) -> KnowledgeView:
    n = sub.n
    return KnowledgeView(access_ms=sub.server_access_ms.copy(), ci=sub.server_ci.copy(),
                         ci_known=np.ones(n, dtype=bool), trust=sub.trust.copy(), caps=sub.caps.copy(),
                         inv_valid=np.ones(n, dtype=bool), up=sub.node_up.copy(), util=sub.utilization().copy(),
                         tenants=sub.tenant_count(), exclusive=sub.exclusive.copy(),
                         state_known=np.ones(n, dtype=bool))


def evaluate_policies(vnfs, kv: KnowledgeView, enabled=POLICIES, unknown: str = "permit") -> dict:
    """Return {policy: (n, N) bool matrix of incompatible pairs} for the enabled policies."""
    n, N = len(vnfs), len(kv.up)
    deny = unknown == "deny"
    out = {}
    if "latency" in enabled:
        m = np.zeros((n, N), dtype=bool)
        for i, v in enumerate(vnfs):
            if "LowLatencyVNF" in v.classes and v.max_access_ms is not None:
                m[i] = kv.access_ms > v.max_access_ms
        out["latency"] = m
    if "capability" in enabled:
        m = np.zeros((n, N), dtype=bool)
        for i, v in enumerate(vnfs):
            if "HardwareAcceleratedVNF" in v.classes and v.cap is not None:
                k = CAPABILITIES.index(v.cap)
                m[i] = (kv.inv_valid & ~kv.caps[:, k]) | (deny & ~kv.inv_valid)
        out["capability"] = m
    if "trust" in enabled:
        req = np.array([v.trust for v in vnfs])
        out["trust"] = kv.trust[None, :] < req[:, None]
    if "isolation" in enabled:
        m = np.zeros((n, N), dtype=bool)
        busy = (kv.state_known & (kv.tenants > 0)) | (deny & ~kv.state_known)
        excl = (kv.state_known & kv.exclusive) | (deny & ~kv.state_known)
        for i, v in enumerate(vnfs):
            m[i] = excl | (busy if "IsolatedVNF" in v.classes else False)
        out["isolation"] = m
    if "opstate" in enabled:
        out["opstate"] = np.repeat(~kv.up[None, :], n, axis=0)
    if "headroom" in enabled:
        m = np.zeros((n, N), dtype=bool)
        hot = (kv.state_known & (kv.util > HEADROOM)) | (deny & ~kv.state_known)
        for i, v in enumerate(vnfs):
            if "MissionCriticalVNF" in v.classes:
                m[i] = hot
        out["headroom"] = m
    if "carbon" in enabled:
        m = np.zeros((n, N), dtype=bool)
        for i, v in enumerate(vnfs):
            if "GreenVNF" in v.classes and v.max_ci is not None:
                m[i] = (kv.ci_known & (kv.ci > v.max_ci)) | (deny & ~kv.ci_known)
        out["carbon"] = m
    return out


def incompatibility(vnfs, kv: KnowledgeView, enabled=POLICIES, unknown: str = "permit") -> tuple[np.ndarray, dict]:
    """I_sem as an (n, N) bool matrix (True = (server, VNF) pair is pruned) plus per-policy provenance."""
    per = evaluate_policies(vnfs, kv, enabled, unknown)
    inc = np.zeros((len(vnfs), len(kv.up)), dtype=bool)
    for m in per.values():
        inc |= m
    return inc, per
