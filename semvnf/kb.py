"""Reference RDF/OWL pipeline (Section 3.3, Fig. 1):

  (1) ABox generation from telemetry, inventory and SFC descriptors (rdflib);
  (2) validation: SHACL shapes (pySHACL) + freshness window -> :isValidated on observations;
  (3) OWL 2 RL closure of TBox+ABox (owlrl), including consistency checking;
  (4) policy rules P1-P7 as SPARQL CONSTRUCT (DL-safe Horn rules) -> :isIncompatibleWith;
  (5) SPARQL extraction of I_sem and of the optimisation parameters.

Each stage is timed separately (Fig. 7) and the number of triples is reported (Fig. 6).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from .catalog import CAPABILITIES
from .rules import POLICIES, KnowledgeView

ONTO_DIR = Path(__file__).resolve().parents[1] / "ontology"
NFV = Namespace("http://example.org/nfv-infra#")
EPOCH0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FRESHNESS_S = 30.0

PREFIX = """PREFIX : <http://example.org/nfv-infra#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
ISVNF = "?v a :VNF ."
ISSRV = "?n a :Server ."

RULES = {
    "latency": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        ?v a :LowLatencyVNF ; :requiresMaxAccessLatency ?lr .
        {ISSRV} ?n :locatedInZone ?z . ?z :hasAccessLatency ?la .
        FILTER (?la > ?lr) }}""",
    "capability": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        ?v a :HardwareAcceleratedVNF ; :requiresCapability ?c .
        {ISSRV} ?n :lacksValidatedCapability ?c . }}""",
    "trust": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        {ISVNF} ?v :requiresTrustLevel ?tr .
        {ISSRV} ?n :belongsToSecurityDomain ?d . ?d :hasTrustLevel ?tn .
        FILTER (?tn < ?tr) }}""",
    "isolation": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        {{ ?v a :IsolatedVNF .
           {ISSRV} ?n :hasResourceState ?s . ?s :isValidated true ; :hasActiveTenantCount ?k .
           FILTER (?k > 0) }}
        UNION
        {{ {ISVNF} ?v :partOfChain ?c .
           {ISSRV} ?n :hasResourceState ?s . ?s :isValidated true ; :hasExclusiveTenant true . }} }}""",
    "opstate": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        {ISVNF} ?v :partOfChain ?c .
        {ISSRV} ?n :hasOperationalState :Unavailable . }}""",
    "headroom": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        ?v a :MissionCriticalVNF .
        {ISSRV} ?n :hasResourceState ?s . ?s :isValidated true ; :hasUtilization ?u .
        FILTER (?u > 0.8) }}""",
    "carbon": f"""CONSTRUCT {{ ?n :isIncompatibleWith ?v }} WHERE {{
        ?v a :GreenVNF ; :requiresMaxCarbonIntensity ?cm .
        {ISSRV} ?n :locatedInZone ?z . ?z :hasCarbonObservation ?o .
        ?o :isValidated true ; :hasCarbonIntensity ?ci .
        FILTER (?ci > ?cm) }}""",
}

EXTRACT_ISEM = PREFIX + """SELECT ?n ?v WHERE { ?n :isIncompatibleWith ?v . }"""
EXTRACT_SERVERS = PREFIX + """SELECT ?n ?cpu ?mem ?acpu ?amem ?pi ?pm WHERE {
    ?n a :Server ; :hasCPUCapacity ?cpu ; :hasMemoryCapacity ?mem ;
       :hasResourceState ?s ; :hasEnergyProfile ?e .
    ?s :hasAvailableCPU ?acpu ; :hasAvailableMemory ?amem .
    ?e :hasIdlePower ?pi ; :hasMaximumPower ?pm . }"""
EXTRACT_LINKS = PREFIX + """SELECT ?l ?u ?w ?bw ?lat WHERE {
    ?l a :SubstrateLink ; :hasSourceNode ?u ; :hasDestinationNode ?w ;
       :hasAvailableBandwidth ?bw ; :hasPropagationLatency ?lat . }"""


def _dec(x) -> Literal:
    return Literal(f"{float(x):.6f}", datatype=XSD.decimal)


def _ts(seconds: float) -> Literal:
    return Literal(EPOCH0 + timedelta(seconds=float(seconds)), datatype=XSD.dateTime)


_TBOX_CACHE: dict = {}


def tbox() -> Graph:
    if "g" not in _TBOX_CACHE:
        g = Graph()
        g.parse(ONTO_DIR / "nfv-infra.ttl", format="turtle")
        _TBOX_CACHE["g"] = g
        sh = Graph()
        sh.parse(ONTO_DIR / "nfv-shapes.ttl", format="turtle")
        _TBOX_CACHE["shapes"] = sh
    return _TBOX_CACHE["g"]


def shapes() -> Graph:
    tbox()
    return _TBOX_CACHE["shapes"]


def build_abox(sub, kv: KnowledgeView, reqs, now: float = 0.0) -> Graph:
    """Materialise the ABox of one placement epoch from the orchestrator's knowledge ``kv``."""
    g = Graph()
    g.bind("", NFV)
    n = sub.n
    res_cpu, res_mem = sub.res_cpu(), sub.res_mem()
    for t in sorted(set(int(x) for x in sub.trust)):
        d = NFV[f"d{t}"]
        g.add((d, RDF.type, NFV.SecurityDomain))
        g.add((d, NFV.hasTrustLevel, Literal(int(t), datatype=XSD.integer)))
    regions = sub.topo.region
    for r in range(sub.topo.n_regions):
        o = NFV[f"co{r}"]
        g.add((o, RDF.type, NFV.CarbonIntensityObservation))
        g.add((o, NFV.hasCarbonIntensity, _dec(sub.region_ci[r])))
    for z in sorted(set(int(x) for x in sub.zone)):          # zones that contain servers
        zz = NFV[f"z{z}"]
        g.add((zz, RDF.type, NFV.GeographicalZone))
        g.add((zz, NFV.hasCarbonObservation, NFV[f"co{z // 2}"]))
    zone_acc = {}
    zone_ci_known = {}
    for j in range(n):
        zone_acc[int(sub.zone[j])] = kv.access_ms[j]
        zone_ci_known[int(sub.zone[j]) // 2] = zone_ci_known.get(int(sub.zone[j]) // 2, True) and bool(kv.ci_known[j])
    for z, acc in zone_acc.items():
        g.add((NFV[f"z{z}"], NFV.hasAccessLatency, _dec(acc)))
    for r in range(sub.topo.n_regions):
        fresh = zone_ci_known.get(r, True)
        g.add((NFV[f"co{r}"], NFV.hasObservationTime, _ts(now if fresh else now - 10 * FRESHNESS_S)))
    for j in range(n):
        s = NFV[f"s{j}"]
        g.add((s, RDF.type, NFV.CoreServer if sub.core[j] else NFV.EdgeServer))
        g.add((s, NFV.locatedInZone, NFV[f"z{int(sub.zone[j])}"]))
        g.add((s, NFV.belongsToSecurityDomain, NFV[f"d{int(kv.trust[j])}"]))
        g.add((s, NFV.hasCPUCapacity, _dec(sub.cpu_cap[j])))
        g.add((s, NFV.hasMemoryCapacity, _dec(sub.mem_cap[j])))
        g.add((s, NFV.hasProcessingFactor, _dec(sub.proc_factor[j])))
        g.add((s, NFV.hasOperationalState, NFV.Available if kv.up[j] else NFV.Unavailable))
        for k, c in enumerate(CAPABILITIES):
            if kv.caps[j, k]:
                g.add((s, NFV.hasCapability, NFV[c]))
            elif kv.inv_valid[j]:
                g.add((s, NFV.lacksValidatedCapability, NFV[c]))
        e = NFV[f"ep{j}"]
        g.add((s, NFV.hasEnergyProfile, e))
        g.add((e, RDF.type, NFV.EnergyProfile))
        g.add((e, NFV.hasIdlePower, _dec(sub.p_idle[j])))
        g.add((e, NFV.hasMaximumPower, _dec(sub.p_max[j])))
        st = NFV[f"rs{j}"]
        g.add((s, NFV.hasResourceState, st))
        g.add((st, RDF.type, NFV.ResourceStateObservation))
        g.add((st, NFV.hasAvailableCPU, _dec(max(0.0, res_cpu[j]))))
        g.add((st, NFV.hasAvailableMemory, _dec(max(0.0, res_mem[j]))))
        g.add((st, NFV.hasUtilization, _dec(min(1.0, max(0.0, kv.util[j])))))
        g.add((st, NFV.hasActiveTenantCount, Literal(int(kv.tenants[j]), datatype=XSD.nonNegativeInteger)))
        g.add((st, NFV.hasExclusiveTenant, Literal(bool(kv.exclusive[j]), datatype=XSD.boolean)))
        g.add((st, NFV.hasObservationTime, _ts(now if kv.state_known[j] else now - 10 * FRESHNESS_S)))
    arcs, res_bw = sub.arcs, sub.res_bw()
    for a in range(len(arcs)):
        if not sub.arc_up[a]:
            continue
        l = NFV[f"l{a}"]
        g.add((l, RDF.type, NFV.SubstrateLink))
        g.add((l, NFV.hasSourceNode, NFV[f"s{int(arcs[a, 0])}"]))
        g.add((l, NFV.hasDestinationNode, NFV[f"s{int(arcs[a, 1])}"]))
        g.add((l, NFV.hasBandwidthCapacity, _dec(sub.arc_cap[a])))
        g.add((l, NFV.hasAvailableBandwidth, _dec(max(0.0, res_bw[a]))))
        g.add((l, NFV.hasPropagationLatency, _dec(sub.arc_prop[a])))
    for req in reqs:
        c = NFV[f"sfc{req.sfc_id}"]
        g.add((c, RDF.type, NFV.ServiceFunctionChain))
        g.add((c, NFV.hasIngress, NFV[f"s{req.ingress}"]))
        g.add((c, NFV.hasEgress, NFV[f"s{req.egress}"]))
        g.add((c, NFV.hasBandwidthDemand, _dec(req.bw)))
        g.add((c, NFV.hasMaxEndToEndLatency, _dec(req.sla_ms)))
        for i, v in enumerate(req.vnfs):
            vv = NFV[f"sfc{req.sfc_id}_v{i}"]
            classes = sorted(v.classes) if v.classes else ["VNF"]
            for cl in classes:
                g.add((vv, RDF.type, NFV[cl]))
            g.add((vv, NFV.partOfChain, c))
            g.add((vv, NFV.hasChainPosition, Literal(i, datatype=XSD.nonNegativeInteger)))
            g.add((vv, NFV.requiresCPU, _dec(v.cpu)))
            g.add((vv, NFV.requiresMemory, _dec(v.mem)))
            g.add((vv, NFV.hasProcessingDelay, _dec(v.proc)))
            g.add((vv, NFV.requiresTrustLevel, Literal(int(v.trust), datatype=XSD.integer)))
            if v.cap is not None:
                g.add((vv, NFV.requiresCapability, NFV[v.cap]))
            if v.max_access_ms is not None:
                g.add((vv, NFV.requiresMaxAccessLatency, _dec(v.max_access_ms)))
            if v.max_ci is not None:
                g.add((vv, NFV.requiresMaxCarbonIntensity, _dec(v.max_ci)))
    return g


def validate(g: Graph, now: float, use_shacl: bool = True) -> dict:
    """SHACL validation + freshness; marks conforming fresh observations with :isValidated true."""
    bad = set()
    conforms = True
    if use_shacl:
        from pyshacl import validate as shacl_validate
        conforms, report_g, _ = shacl_validate(g, shacl_graph=shapes(), inference="none", advanced=False,
                                               allow_warnings=True)
        SH = Namespace("http://www.w3.org/ns/shacl#")
        for res in report_g.subjects(RDF.type, SH.ValidationResult):
            fn = report_g.value(res, SH.focusNode)
            if fn is not None:
                bad.add(fn)
    limit = EPOCH0 + timedelta(seconds=float(now) - FRESHNESS_S)
    n_val = 0
    for o in list(g.subjects(RDF.type, NFV.ResourceStateObservation)) + list(g.subjects(RDF.type, NFV.CarbonIntensityObservation)):
        t = g.value(o, NFV.hasObservationTime)
        ok = o not in bad and t is not None and t.toPython() >= limit
        if ok:
            g.add((o, NFV.isValidated, Literal(True)))
            n_val += 1
    return dict(conforms=bool(conforms), n_violations=len(bad), n_validated=n_val)


def closure(g: Graph, mode: str = "owlrl") -> dict:
    """OWL 2 RL (or RDFS) deductive closure of TBox + ABox, in place."""
    import owlrl
    before = len(g)
    for t in tbox():
        g.add(t)
    if mode == "owlrl":
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics, axiomatic_triples=False, datatype_axioms=False).expand(g)
    elif mode == "rdfs":
        owlrl.DeductiveClosure(owlrl.RDFS_Semantics, axiomatic_triples=False, datatype_axioms=False).expand(g)
    # OWL-RL inconsistency: individuals typed owl:Nothing
    from rdflib import OWL
    inconsistent = len(list(g.subjects(RDF.type, OWL.Nothing)))
    return dict(triples_before=before, triples_after=len(g), inconsistent=inconsistent)


def apply_rules(g: Graph, enabled=POLICIES, timing: dict | None = None) -> dict:
    """Execute the policy rules; returns {policy: set((server_iri, vnf_iri))} and adds the conclusions."""
    out = {}
    for p in POLICIES:
        if p not in enabled:
            continue
        t0 = time.perf_counter()
        res = g.query(PREFIX + RULES[p])
        pairs = set((s, o) for (s, _, o) in res)
        if timing is not None:
            timing[p] = time.perf_counter() - t0
        out[p] = pairs
    for pairs in out.values():
        for s, o in pairs:
            g.add((s, NFV.isIncompatibleWith, o))
    return out


def _idx(iri, prefix: str) -> int:
    return int(str(iri).split("#")[-1][len(prefix):])


def extract(g: Graph, reqs, n_servers: int) -> tuple[dict, dict]:
    """SPARQL extraction of I_sem per request and of the numeric optimisation parameters."""
    isem = {r.sfc_id: np.zeros((r.n, n_servers), dtype=bool) for r in reqs}
    for row in g.query(EXTRACT_ISEM):
        n_iri, v_iri = row
        name = str(v_iri).split("#")[-1]          # sfc{id}_v{i}
        sid, vi = name[3:].split("_v")
        isem[int(sid)][int(vi), _idx(n_iri, "s")] = True
    params = dict(servers=len(list(g.query(EXTRACT_SERVERS))), links=len(list(g.query(EXTRACT_LINKS))))
    return isem, params


def run_pipeline(sub, kv: KnowledgeView, reqs, now: float = 0.0, enabled=POLICIES,
                 closure_mode: str = "owlrl", use_shacl: bool = True) -> dict:
    """Full reference pipeline with per-stage timing; returns I_sem per request."""
    t0 = time.perf_counter()
    g = build_abox(sub, kv, reqs, now)
    t1 = time.perf_counter()
    abox_triples = len(g)
    val = validate(g, now, use_shacl=use_shacl)
    t2 = time.perf_counter()
    cl = closure(g, closure_mode) if closure_mode != "none" else dict(triples_after=len(g), inconsistent=0)
    t3 = time.perf_counter()
    rule_t = {}
    per = apply_rules(g, enabled, rule_t)
    t4 = time.perf_counter()
    isem, params = extract(g, reqs, sub.n)
    t5 = time.perf_counter()
    return dict(isem=isem, per_policy={k: len(v) for k, v in per.items()}, rule_time=rule_t, abox_triples=abox_triples,
                tbox_triples=len(tbox()), closure_triples=cl["triples_after"], inconsistent=cl["inconsistent"],
                validation=val, params=params,
                t_abox=t1 - t0, t_validate=t2 - t1, t_closure=t3 - t2, t_rules=t4 - t3, t_extract=t5 - t4,
                t_total=t5 - t0)
