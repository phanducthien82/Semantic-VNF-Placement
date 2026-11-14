"""Core consistency tests (run: python -m pytest tests  or  python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semvnf import ga, ilp, kb  # noqa: E402
from semvnf.model import Instance, remove_cycles, validate  # noqa: E402
from semvnf.routing import Router  # noqa: E402
from semvnf.rules import incompatibility, view_from_substrate  # noqa: E402
from semvnf.substrate import make_substrate  # noqa: E402
from semvnf.topology import load_topology  # noqa: E402
from semvnf.workload import WorkloadConfig, generate_trace  # noqa: E402
from semvnf.rules import POLICIES, evaluate_policies  # noqa: E402
from semvnf import compiler  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_rules import evaluate_policies_manual  # noqa: E402


def _loaded(name="euro28", seed=3):
    topo = load_topology(name)
    sub = make_substrate(topo, seed)
    rng = np.random.default_rng(seed)
    sub.cpu_used = np.floor(rng.uniform(0, 0.9, sub.n) * sub.cpu_cap)
    sub.n_vnfs = (sub.cpu_used > 0).astype(int)
    for j in range(sub.n):
        for t in range(int(rng.integers(0, 3))):
            sub.tenants[j][10_000 + t] += 1
    sub.exclusive[rng.choice(sub.n, 2, replace=False)] = True
    sub.node_up[rng.choice(sub.n, 1, replace=False)] = False
    return sub


def test_rdf_equals_compiled():
    sub = _loaded()
    trace = generate_trace(sub, 3, WorkloadConfig(n_arrivals=8))
    kv = view_from_substrate(sub)
    kv.state_known[:3] = False
    kv.inv_valid[3:6] = False
    res = kb.run_pipeline(sub, kv, trace, now=50.0)
    for r in trace:
        assert np.array_equal(res["isem"][r.sfc_id], incompatibility(r.vnfs, kv)[0])
    assert res["inconsistent"] == 0


def test_conservative_mode_prunes_unknown_pairs():
    """unknown='deny' prunes a superset of unknown='permit'; both coincide when everything is known,
    and every pair that only 'deny' prunes involves an unknown attribute."""
    sub = _loaded()
    trace = generate_trace(sub, 3, WorkloadConfig(n_arrivals=8))
    full = view_from_substrate(sub)
    kv = view_from_substrate(sub)
    kv.state_known[:3] = False
    kv.inv_valid[3:6] = False
    kv.ci_known[6:9] = False
    unknown_srv = ~kv.state_known | ~kv.inv_valid | ~kv.ci_known
    for r in trace:
        permit = incompatibility(r.vnfs, kv, unknown="permit")[0]
        deny = incompatibility(r.vnfs, kv, unknown="deny")[0]
        assert np.all(deny >= permit)
        assert not np.any((deny & ~permit)[:, ~unknown_srv])
        assert np.array_equal(incompatibility(r.vnfs, full, unknown="deny")[0],
                              incompatibility(r.vnfs, full, unknown="permit")[0])


def test_ilp_objective_and_validation():
    sub = _loaded("germany50", 5)
    trace = generate_trace(sub, 5, WorkloadConfig(n_arrivals=10))
    for req in trace:
        kv = view_from_substrate(sub)
        inc, _ = incompatibility(req.vnfs, kv)
        inst = Instance(req, sub, ~inc, kv=kv)
        res = ilp.solve(inst)
        if res["servers"] is None:
            continue
        ok, why = validate(inst, res["servers"], res["paths"])
        assert ok, why
        assert abs(inst.evaluate(res["servers"], res["paths"])["J"] - res["obj"]) < 1e-6 * max(1.0, abs(res["obj"]))
        assert res["cycles_removed"] == 0


def test_ga_feasible_solutions_are_valid_and_bounded():
    sub = _loaded("euro28", 7)
    router = Router(sub.n, sub.arcs, sub.arc_prop + sub.arc_tx, sub.usable_arcs(), k=5)
    trace = generate_trace(sub, 7, WorkloadConfig(n_arrivals=10))
    for req in trace:
        kv = view_from_substrate(sub)
        inc, _ = incompatibility(req.vnfs, kv)
        inst = Instance(req, sub, ~inc, kv=kv)
        res = ga.solve(inst, router, ga.GAConfig(seed=1))
        if res["status"] == "feasible":
            ok, why = validate(inst, res["servers"], res["paths"])
            assert ok, why
            assert res["best"]["J"] < inst.J_upper_bound()


def test_remove_cycles():
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 1], [4, 5], [5, 4]])
    # path 0-1-2 plus a cycle 1-2-3-1 sharing nodes, plus a detached cycle 4-5-4
    p = remove_cycles([0, 1, 2, 3, 4, 5], arcs, 0, 2)
    assert p == [0, 1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_compiled_rules_equal_manual_reference():
    """The predicates generated from policies.rq agree with the hand-written transcription of
    P1-P7 on random knowledge views, in the default and in the conservative mode."""
    rng = np.random.default_rng(1)
    checked = 0
    for name, seed in (("euro28", 1), ("germany50", 2)):
        sub = _loaded(name, seed)
        trace = generate_trace(sub, seed, WorkloadConfig(n_arrivals=30))
        for _ in range(4):
            kv = view_from_substrate(sub)
            kv.state_known &= rng.random(sub.n) > 0.3
            kv.inv_valid &= rng.random(sub.n) > 0.3
            kv.ci_known &= rng.random(sub.n) > 0.3
            kv.util = rng.uniform(0, 1, sub.n)
            kv.trust = rng.integers(1, 4, sub.n)
            for r in trace:
                for mode in ("permit", "deny"):
                    a = evaluate_policies(r.vnfs, kv, POLICIES, mode)
                    b = evaluate_policies_manual(r.vnfs, kv, POLICIES, mode)
                    for pol in POLICIES:
                        assert np.array_equal(a[pol], b[pol]), (name, r.sfc_id, pol, mode)
                        checked += 1
    assert checked > 1000


def test_compiler_rejects_rules_outside_the_class():
    """Rules outside the supported class are rejected at compile time, not silently mis-compiled."""
    bad = {
        "two_vnfs": """CONSTRUCT { ?n :isIncompatibleWith ?v } WHERE {
            ?v a :VNF ; :partOfChain ?c . ?w a :VNF ; :partOfChain ?c . ?n a :Server .
            ?n :hostsVNF ?w . }""",
        "aggregate": """CONSTRUCT { ?n :isIncompatibleWith ?v } WHERE {
            ?v a :VNF . ?n a :Server . ?n :locatedInZone ?z .
            { SELECT ?z (COUNT(?m) AS ?k) WHERE { ?m :locatedInZone ?z } GROUP BY ?z }
            FILTER (?k > 3) }""",
        "unbound_path": """CONSTRUCT { ?n :isIncompatibleWith ?v } WHERE {
            ?v a :VNF . ?n a :Server . ?n :hasEnergyProfile ?e . ?e :hasIdlePower ?p . FILTER (?p > 100) }""",
    }
    for name, text in bad.items():
        try:
            compiler.Program({name: text})
        except compiler.UnsupportedRule:
            continue
        raise AssertionError(f"rule {name} should have been rejected")
    # a new rule inside the class compiles without any code change
    extra = """CONSTRUCT { ?n :isIncompatibleWith ?v } WHERE {
        ?v a :MissionCriticalVNF . ?n a :Server . ?n :belongsToSecurityDomain ?d . ?d :hasTrustLevel ?t .
        ?n :hasResourceState ?s . ?s :isValidated true ; :hasUtilization ?u .
        FILTER (?t < 2 && ?u > 0.5) }"""
    prog = compiler.Program({"mc_trust": extra})
    sub = _loaded()
    kv = view_from_substrate(sub)
    trace = generate_trace(sub, 3, WorkloadConfig(n_arrivals=5))
    for r in trace:
        m = prog.evaluate(r.vnfs, kv, ("mc_trust",))["mc_trust"]
        mc = np.array(["MissionCriticalVNF" in v.classes for v in r.vnfs])
        expect = mc[:, None] & ((kv.trust < 2) & kv.state_known & (kv.util > 0.5))[None, :]
        assert np.array_equal(m, expect)


def test_shacl_guards_functional_object_properties():
    """Two values of a functional object property are not an OWL 2 RL inconsistency (no unique-name
    assumption); the SHACL cardinality constraint withholds the server from the rule evaluation.
    Two operational states, which are declared owl:differentFrom, are an inconsistency."""
    from rdflib import OWL
    sub = _loaded()
    trace = generate_trace(sub, 3, WorkloadConfig(n_arrivals=4))
    kv = view_from_substrate(sub)
    zones = sorted(set(int(z) for z in sub.zone))
    j = 0
    other = next(z for z in zones if z != int(sub.zone[j]))
    # (a) a second zone for server 0, no SHACL: the OWL 2 RL closure infers sameAs, no error
    g = kb.build_abox(sub, kv, trace, now=50.0)
    g.add((kb.NFV[f"s{j}"], kb.NFV.locatedInZone, kb.NFV[f"z{other}"]))
    kb.validate(g, 50.0, use_shacl=False)
    cl = kb.closure(g)
    assert cl["inconsistent"] == 0
    assert (kb.NFV[f"z{int(sub.zone[j])}"], OWL.sameAs, kb.NFV[f"z{other}"]) in g
    # (b) with SHACL the server is withheld: no rule can conclude anything about it
    res = kb.run_pipeline(sub, kv, trace, now=50.0)
    g2 = kb.build_abox(sub, kv, trace, now=50.0)
    g2.add((kb.NFV[f"s{j}"], kb.NFV.locatedInZone, kb.NFV[f"z{other}"]))
    val = kb.validate(g2, 50.0)
    assert not val["conforms"] and val["n_withheld"] >= 1
    kb.closure(g2)
    kb.apply_rules(g2)
    isem, _ = kb.extract(g2, trace, sub.n)
    for r in trace:
        assert not isem[r.sfc_id][:, j].any()
        ref = res["isem"][r.sfc_id].copy()
        ref[:, j] = False
        assert np.array_equal(isem[r.sfc_id], ref)
    # (c) two operational states: Available and Unavailable are declared different -> inconsistency
    g3 = kb.build_abox(sub, kv, trace, now=50.0)
    g3.add((kb.NFV[f"s{j}"], kb.NFV.hasOperationalState, kb.NFV.Unavailable))
    kb.validate(g3, 50.0, use_shacl=False)
    cl3 = kb.closure(g3)
    assert cl3["inconsistent"] >= 1
    # (d) disjoint classes with a common individual are an inconsistency as well
    g4 = kb.build_abox(sub, kv, trace, now=50.0)
    g4.add((kb.NFV[f"s{j}"], kb.kb_RDF_TYPE if hasattr(kb, "kb_RDF_TYPE") else __import__("rdflib").RDF.type,
            kb.NFV.CoreServer if not sub.core[j] else kb.NFV.EdgeServer))
    kb.validate(g4, 50.0, use_shacl=False)
    assert kb.closure(g4)["inconsistent"] >= 1
