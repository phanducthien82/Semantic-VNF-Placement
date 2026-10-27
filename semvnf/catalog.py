"""VNF catalogue and the five reference SFC templates (Table 8 of the revised paper).

Demands are in CPU cores / GB RAM, processing delays in ms (mean per-packet sojourn
time of the VNF at its dimensioned operating point on a core server; edge servers are
25 % slower).  Semantic classes and policy attributes are the ontology classes of
Table 2; they are attached per template so that the same VNF type can carry
different policies in different slices (e.g. a firewall is mission-critical only in
the URLLC slice).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ VNF types
#                 name      cpu  mem  proc_ms  required capability
VNF_TYPES = {
    "FW":    dict(cpu=2, mem=2, proc=0.10, cap=None),
    "NAT":   dict(cpu=1, mem=1, proc=0.05, cap=None),
    "IDS":   dict(cpu=4, mem=4, proc=0.40, cap=None),
    "DPI":   dict(cpu=4, mem=4, proc=0.60, cap="SmartNIC"),
    "LB":    dict(cpu=1, mem=2, proc=0.05, cap=None),
    "TC":    dict(cpu=6, mem=8, proc=1.50, cap="GPU"),          # video transcoder
    "UPF":   dict(cpu=2, mem=4, proc=0.10, cap=None),          # 5G user-plane function
    "VPN":   dict(cpu=2, mem=2, proc=0.20, cap="CryptoAccel"),  # IPsec gateway
    "CACHE": dict(cpu=2, mem=8, proc=0.20, cap=None),
    "IOT":   dict(cpu=2, mem=4, proc=2.00, cap=None),          # IoT analytics / logging
}

CAPABILITIES = ("GPU", "SmartNIC", "CryptoAccel")
EDGE_SLOWDOWN = 1.25


@dataclass(frozen=True)
class VNFSpec:
    vtype: str
    cpu: int
    mem: int
    proc: float
    classes: frozenset            # ontology classes (subclasses of VNF)
    cap: str | None = None        # requiresCapability
    trust: int = 1                # requiresTrustLevel
    max_access_ms: float | None = None   # requiresLatency (LowLatencyVNF)
    max_ci: float | None = None   # requiresMaxCarbonIntensity (GreenVNF)


@dataclass(frozen=True)
class Template:
    name: str
    chain: tuple                  # VNF type names in order
    prob: float                   # share of arrivals
    bw: tuple                     # (lo, hi) Mbps per virtual link
    sla_ms: float                 # end-to-end latency bound (ingress -> egress)
    local: bool                   # True: local breakout (egress = ingress site, MEC application)
    overrides: dict = field(default_factory=dict)   # per-VNF-type policy overrides
    slice_classes: tuple = ()     # classes applied to every VNF of the chain
    slice_trust: int = 1          # minimum trust level of every VNF of the chain


TEMPLATES = (
    Template("Web", ("LB", "CACHE"), 0.25, (50, 200), 50.0, False),
    Template("URLLC", ("FW", "UPF", "IDS"), 0.20, (20, 100), 10.0, True,
             overrides={"UPF": dict(classes={"LowLatencyVNF"}, max_access_ms=2.0),
                        "IDS": dict(trust=2)},
             slice_classes=("MissionCriticalVNF",)),
    Template("mIoT", ("NAT", "FW", "IOT"), 0.20, (10, 50), 150.0, False,
             overrides={"IOT": dict(classes={"GreenVNF"}, max_ci=250.0)},
             slice_classes=("DelayTolerantVNF",)),
    Template("Enterprise", ("VPN", "FW", "IDS", "LB"), 0.10, (50, 200), 40.0, False,
             overrides={"VPN": dict(classes={"IsolatedVNF"}, trust=3)},
             slice_trust=2),
    Template("eMBB", ("FW", "NAT", "DPI", "TC", "CACHE"), 0.25, (200, 600), 50.0, False),
)
TEMPLATE_BY_NAME = {t.name: t for t in TEMPLATES}


def vnf_specs(template: Template) -> list[VNFSpec]:
    """Instantiate the policy-annotated VNF specifications of a template."""
    out = []
    for vt in template.chain:
        base = VNF_TYPES[vt]
        ov = template.overrides.get(vt, {})
        classes = set(template.slice_classes) | set(ov.get("classes", ()))
        if base["cap"] is not None:
            classes.add("HardwareAcceleratedVNF")
        trust = max(template.slice_trust, ov.get("trust", 1))
        out.append(VNFSpec(vtype=vt, cpu=base["cpu"], mem=base["mem"], proc=base["proc"],
                           classes=frozenset(classes), cap=base["cap"], trust=trust,
                           max_access_ms=ov.get("max_access_ms"), max_ci=ov.get("max_ci")))
    return out


def mean_cpu_per_sfc() -> float:
    return sum(t.prob * sum(VNF_TYPES[v]["cpu"] for v in t.chain) for t in TEMPLATES)


def mean_chain_length() -> float:
    return sum(t.prob * len(t.chain) for t in TEMPLATES)
