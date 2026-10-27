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
