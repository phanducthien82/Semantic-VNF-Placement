"""Imperfect-knowledge model for the robustness study (Section 4.9, reply to R2 comment 6).

The orchestrator reasons over an *observed* knowledge view that may differ from the ground
truth used by the validator.  With rate ``rate`` per server (or zone) the following errors are
injected (kind='mixed' applies all of them):

  stale    -- resource-state observation older than the freshness window (unknown -> no pruning);
  missing  -- capability inventory not validated (unknown -> no pruning);
  carbon   -- carbon-intensity observation not validated (unknown -> no pruning);
  wrong    -- incorrect assertions: trust level off by one, one capability flipped,
              utilisation off by U(-0.2, 0.2) (can create false positives and false negatives).

Unknown attributes never create pruning (three-valued semantics), so they can only produce
false negatives (policy violations caught by the validator); wrong assertions can produce both.
"""
from __future__ import annotations

import copy

import numpy as np


def perturb(kv, sub, cfg: dict, rng: np.random.Generator):
    rate = float(cfg.get("rate", 0.0))
    kind = cfg.get("kind", "mixed")
    obs = copy.deepcopy(kv)
    n = len(obs.up)
    kinds = ("stale", "missing", "carbon", "wrong") if kind == "mixed" else (kind,)
    if "stale" in kinds:
        obs.state_known &= ~(rng.random(n) < rate)
    if "missing" in kinds:
        obs.inv_valid &= ~(rng.random(n) < rate)
    if "carbon" in kinds:
        reg = sub.topo.region
        bad = rng.random(sub.topo.n_regions) < rate
        obs.ci_known &= ~bad[reg]
    if "wrong" in kinds:
        m = rng.random(n) < rate
        obs.trust = np.where(m, np.clip(obs.trust + rng.choice([-1, 1], n), 1, 3), obs.trust)
        m = rng.random(n) < rate
        k = rng.integers(obs.caps.shape[1], size=n)
        caps = obs.caps.copy()
        caps[np.arange(n)[m], k[m]] = ~caps[np.arange(n)[m], k[m]]
        obs.caps = caps
        m = rng.random(n) < rate
        obs.util = np.where(m, np.clip(obs.util + rng.uniform(-0.2, 0.2, n), 0, 1), obs.util)
    return obs
