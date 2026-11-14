"""Hand-written reference implementation of the seven policy rules (test oracle only).

This is the manual transcription of rules P1-P7 that the online simulator used before the rule
compiler (semvnf/compiler.py) was introduced.  It is kept only to cross-check the compiled
predicates: tests/test_core.py verifies that both agree on random knowledge views in both modes.
"""
from __future__ import annotations

import numpy as np

from semvnf.catalog import CAPABILITIES
from semvnf.rules import HEADROOM, POLICIES, KnowledgeView


def evaluate_policies_manual(vnfs, kv: KnowledgeView, enabled=POLICIES, unknown: str = "permit") -> dict:
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
