"""Rule compiler: SPARQL CONSTRUCT policy rules -> vectorised predicates (Section 3.6).

The online predicates are *generated* from the same rule text that the reference RDF/OWL pipeline
executes (``ontology/policies.rq``); no individual policy is transcribed by hand.  The only
hand-written part is the *schema binding* below, which maps property paths of the ``nfv-infra``
ontology to the arrays of the placement snapshot (the knowledge view of ``rules.py``), i.e. the
mapping of Table 6 of the paper.  For every rule the compiler emits one Python function that
evaluates the rule with NumPy on all (VNF, server) pairs of a request at once; the emitted source
is kept (``Program.source``) so that it can be inspected and tested.

Supported rule class (checked at compile time; anything else raises ``UnsupportedRule``):

* a CONSTRUCT template ``?n :isIncompatibleWith ?v`` with a server variable ``?n`` and a VNF
  variable ``?v``;
* a body that is a basic graph pattern (conjunction of triple patterns), optionally with FILTER
  expressions, or a UNION of such patterns;
* triple patterns of the forms ``?x rdf:type C`` (class atoms, resolved with the subclass axioms of
  the TBox), ``?n p ?y`` / ``?y q ?z`` along a bound property path from the server (zone, security
  domain, validated observations, capability inventory, operational state), ``?v p ?y`` for VNF
  attributes, with ``?y`` either a fresh variable, a constant (individual or literal), or a variable
  already bound on the other side (identity join through a shared individual, e.g. a capability);
* FILTER expressions built from comparisons (``< <= = != >= >``) between bound variables and
  constants, combined with ``&&``, ``||`` and ``!``.

Outside the class (and therefore model-extending in the sense of Section 3.1): aggregation,
recursion, negation as failure, property paths that are not in the binding, rules that relate
several VNFs or several servers, and existential conclusions.

Three-valued semantics: a rule fires only where every validated attribute it uses is known.  In the
conservative mode (``unknown="deny"``) a pair is additionally marked incompatible when the VNF-side
atoms hold and some validated attribute used by the rule is unknown for the server.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdflib import RDF, RDFS, Graph, Literal, URIRef, Variable
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

from .catalog import CAPABILITIES

ONTO_DIR = Path(__file__).resolve().parents[1] / "ontology"
NS = "http://example.org/nfv-infra#"
PREFIX = """PREFIX : <http://example.org/nfv-infra#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


class UnsupportedRule(ValueError):
    """The rule is outside the compilable class."""


# ---------------------------------------------------------------------------------------- rule texts
def load_rule_texts(path: Path = ONTO_DIR / "policies.rq") -> dict:
    """Parse ``policies.rq`` into {policy name: CONSTRUCT text without PREFIX lines}, in file order."""
    out, name, buf = {}, None, []

    def flush():
        if name is not None:
            body = "\n".join(l for l in buf if not l.startswith("PREFIX") and not l.startswith("#")).strip()
            if body:
                out[name] = body

    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#\s*----\s*(\w+)", line)
        if m:
            flush()
            name, buf = m.group(1), []
        else:
            buf.append(line)
    flush()
    return out


RULE_TEXTS = load_rule_texts()


# ---------------------------------------------------------------------------------------- schema binding
@dataclass(frozen=True)
class Column:
    """A per-server quantity reachable from the server variable along a property path.

    ``get(kv)`` returns an (N,) array.  ``validity`` marks the boolean flags that encode "the
    observation is validated" (``:isValidated``); they define what "unknown" means for the
    conservative mode.
    """
    name: str
    get: callable
    validity: bool = False
    codes: tuple = ()           # individuals of a finite vocabulary coded as booleans: ((name, value), ...)
    boolean: bool = False       # the array is boolean (flags): comparisons with true/false are the array itself


@dataclass(frozen=True)
class SetColumn:
    """A set-valued per-server property over a finite vocabulary of individuals.

    ``get(kv)`` returns an (N, |vocabulary|) boolean matrix (True = the triple is asserted);
    ``known(kv)`` returns the (N,) validity mask under which the assertions are complete.
    """
    name: str
    vocabulary: tuple
    get: callable
    known: callable = None


# property paths from the server variable (local names) -> columns of the knowledge view
SERVER_PATHS = {
    ("locatedInZone", "hasAccessLatency"): Column("access_ms", lambda kv: kv.access_ms),
    ("locatedInZone", "hasCarbonObservation", "isValidated"): Column("ci_known", lambda kv: kv.ci_known, validity=True,
                                                                       boolean=True),
    ("locatedInZone", "hasCarbonObservation", "hasCarbonIntensity"): Column("ci", lambda kv: kv.ci),
    ("belongsToSecurityDomain", "hasTrustLevel"): Column("trust", lambda kv: kv.trust),
    ("hasOperationalState",): Column("up", lambda kv: kv.up, codes=(("Available", True), ("Unavailable", False)),
                                     boolean=True),
    ("hasResourceState", "isValidated"): Column("state_known", lambda kv: kv.state_known, validity=True, boolean=True),
    ("hasResourceState", "hasUtilization"): Column("util", lambda kv: kv.util),
    ("hasResourceState", "hasActiveTenantCount"): Column("tenants", lambda kv: kv.tenants),
    ("hasResourceState", "hasExclusiveTenant"): Column("exclusive", lambda kv: kv.exclusive, boolean=True),
    ("hasCapability",): SetColumn("caps", CAPABILITIES, lambda kv: kv.caps),
    ("lacksValidatedCapability",): SetColumn("lacks", CAPABILITIES, lambda kv: kv.inv_valid[:, None] & ~kv.caps,
                                             known=lambda kv: kv.inv_valid),
}
# intermediate nodes of the paths above (zone, domain, observation ...) are implied by the prefixes
SERVER_NODES = {p[:k] for p in SERVER_PATHS for k in range(1, len(p))}
SERVER_CLASSES = {"Server"}                       # every server of the knowledge view

# VNF properties -> attribute of the VNF specification; None = pattern always bound (no attribute)
VNF_PROPS = {
    "requiresMaxAccessLatency": "max_access_ms",
    "requiresTrustLevel": "trust",
    "requiresCapability": "cap",
    "requiresMaxCarbonIntensity": "max_ci",
    "requiresCPU": "cpu",
    "requiresMemory": "mem",
    "hasProcessingDelay": "proc",
    "partOfChain": None,                          # every VNF belongs to a chain
}


def _local(term) -> str:
    return str(term).split("#")[-1]


def _subclass_closure(tbox_path: Path = ONTO_DIR / "nfv-infra.ttl") -> dict:
    """{class local name: set of its superclasses (transitive), including itself}."""
    g = Graph()
    g.parse(tbox_path, format="turtle")
    sup = {}
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            sup.setdefault(_local(s), set()).add(_local(o))
    out = {}
    for c in list(g.subjects(RDF.type, URIRef("http://www.w3.org/2002/07/owl#Class"))):
        name = _local(c)
        cl, todo = {name}, [name]
        while todo:
            x = todo.pop()
            for y in sup.get(x, ()):
                if y not in cl:
                    cl.add(y)
                    todo.append(y)
        out[name] = cl
    return out


def _const(term):
    if isinstance(term, Literal):
        val = term.toPython()
        if isinstance(val, bool):
            return val
        try:
            return float(val)
        except (TypeError, ValueError):
            return str(val)
    return _local(term)


# ---------------------------------------------------------------------------------------- analysed form
@dataclass
class Pattern:
    """One basic graph pattern (plus its filters) of a rule, after analysis."""
    vnf_classes: list = field(default_factory=list)       # class names required of the VNF
    vnf_bound: list = field(default_factory=list)         # VNF attributes that must be present
    vnf_consts: list = field(default_factory=list)        # (attribute, constant) equalities
    server_consts: list = field(default_factory=list)     # (column, constant) equalities
    joins: list = field(default_factory=list)             # (column, VNF attribute) identity joins
    validity: list = field(default_factory=list)          # validity columns / set columns used
    filters: list = field(default_factory=list)           # FILTER expression trees
    bindings: dict = field(default_factory=dict)          # variable -> ("server", column) | ("vnf", attr)


# ---------------------------------------------------------------------------------------- runtime views
class _VnfView:
    """Per-request view of the VNFs: class masks and attribute arrays.

    VNF specifications are frozen values, so everything that depends only on the chain is cached
    in ``cache`` under the key ``tuple(vnfs)``; for a recurring chain the VNF side of a rule costs
    one dictionary lookup per arrival.
    """
    __slots__ = ("vnfs", "n", "key", "cache", "_closure")

    def __init__(self, vnfs, closure, cache):
        self.vnfs, self.n, self._closure, self.cache = vnfs, len(vnfs), closure, cache
        self.key = tuple(vnfs)

    def pattern(self, name, k, fn):
        """(vnf_ok, non-empty) of pattern k of rule ``name``; ``fn(V)`` computes vnf_ok on a cache miss."""
        ck = (self.key, name, k)
        r = self.cache.get(ck)
        if r is None:
            m = fn(self)
            r = (m, bool(m.any()))
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[ck] = r
        return r

    def cls(self, c):
        return np.fromiter((c in self._closure(v) for v in self.vnfs), dtype=bool, count=self.n)

    def has(self, a):
        return np.fromiter((getattr(v, a) is not None for v in self.vnfs), dtype=bool, count=self.n)

    def num(self, a):
        ck = (self.key, "num", a)
        m = self.cache.get(ck)
        if m is None:
            m = np.array([np.nan if (x := getattr(v, a)) is None else float(x) for v in self.vnfs])
            self.cache[ck] = m
        return m

    def obj(self, a):
        ck = (self.key, "obj", a)
        m = self.cache.get(ck)
        if m is None:
            m = np.array([getattr(v, a) for v in self.vnfs], dtype=object)
            self.cache[ck] = m
        return m

    def idx(self, a, vocabulary):
        ck = (self.key, "idx", a)
        m = self.cache.get(ck)
        if m is None:
            m = np.fromiter(((vocabulary.index(x) if x in vocabulary else -1) for x in self.obj(a)), dtype=int,
                            count=self.n)
            self.cache[ck] = m
        return m


class _SrvView:
    """Per-epoch view of the servers: columns of the knowledge view, computed lazily and cached."""
    __slots__ = ("kv", "N", "_cache")

    def __init__(self, kv):
        self.kv, self.N, self._cache = kv, len(kv.up), {}

    def col(self, column):
        m = self._cache.get(column.name)
        if m is None:
            m = column.get(self.kv)
            self._cache[column.name] = m
        return m

    def join(self, column, idx):
        """(n, N) matrix: pair (i, j) holds the individual idx[i] of the set-valued column (False if idx[i] < 0)."""
        mat = self.col(column)
        out = np.zeros((len(idx), self.N), dtype=bool)
        ok = idx >= 0
        if ok.any():
            out[ok] = mat[:, idx[ok]].T
        return out


# ---------------------------------------------------------------------------------------- the compiler
class Program:
    """The compiled rule set: ``evaluate(vnfs, kv, enabled, unknown)`` -> {policy: (n, N) bool}."""

    def __init__(self, texts: dict):
        self.classes = _subclass_closure()
        self._closure_cache = {}
        self._vcache = {}           # chain-level cache of the VNF side of the patterns
        self.patterns = {}          # rule name -> [Pattern]
        self.source = {}            # rule name -> emitted Python source
        self.functions = {}         # rule name -> compiled function
        self.columns = []           # column objects referenced by the emitted code (C[k])
        for name, text in texts.items():
            self.patterns[name] = self._analyse(name, text)
            self.source[name] = self._emit(name, self.patterns[name])
        ns = {"np": np, "C": self.columns}
        exec("\n".join(self.source.values()), ns)
        for name in texts:
            self.functions[name] = ns[f"rule_{name}"]

    # ------------------------------------------------------------------ analysis
    def _analyse(self, name: str, text: str) -> list:
        q = prepareQuery(PREFIX + text)
        alg = q.algebra
        if alg.name != "ConstructQuery" or len(alg.template) != 1:
            raise UnsupportedRule(f"{name}: expected a CONSTRUCT with one template triple")
        n_var, pred, v_var = alg.template[0]
        if _local(pred) != "isIncompatibleWith" or not isinstance(n_var, Variable) or not isinstance(v_var, Variable):
            raise UnsupportedRule(f"{name}: the conclusion must be isIncompatibleWith(?server, ?vnf)")
        node = alg.p
        if node.name == "Project":
            node = node.p
        return [self._analyse_bgp(name, bgp, filters, n_var, v_var) for bgp, filters in self._branches(name, node)]

    def _branches(self, name, node):
        """Flatten UNIONs into (triples, [filter expressions]) branches."""
        if node.name == "Union":
            return self._branches(name, node.p1) + self._branches(name, node.p2)
        if node.name == "Filter":
            inner = self._branches(name, node.p)
            if len(inner) != 1:
                raise UnsupportedRule(f"{name}: FILTER over a UNION is not supported")
            triples, filters = inner[0]
            return [(triples, filters + [node.expr])]
        if node.name == "BGP":
            return [(list(node.triples), [])]
        raise UnsupportedRule(f"{name}: unsupported algebra node {node.name}")

    def _analyse_bgp(self, name, triples, filters, n_var, v_var) -> Pattern:
        P = Pattern()
        paths = {n_var: ()}                     # node variable -> property path from the server
        pending = list(triples)
        progress = True
        while pending and progress:
            progress = False
            for t in list(pending):
                s, p, o = t
                if p == RDF.type:
                    cls = _local(o)
                    if s == n_var:
                        if cls not in SERVER_CLASSES:
                            raise UnsupportedRule(f"{name}: server class {cls} is not bound")
                    elif s == v_var:
                        if cls not in self.classes:
                            raise UnsupportedRule(f"{name}: unknown class {cls}")
                        P.vnf_classes.append(cls)
                    elif s in paths:
                        pass                    # type atom of an intermediate node (zone, observation ...)
                    else:
                        continue                # subject not bound yet
                    pending.remove(t)
                    progress = True
                    continue
                pl = _local(p)
                if s == v_var:                  # VNF attribute
                    if pl not in VNF_PROPS:
                        raise UnsupportedRule(f"{name}: VNF property {pl} is not bound")
                    attr = VNF_PROPS[pl]
                    if isinstance(o, Variable):
                        if o in P.bindings and P.bindings[o][0] == "server":
                            col = P.bindings[o][1]
                            P.joins.append((col, attr))
                            if isinstance(col, SetColumn) or col.validity:
                                P.validity.append(col)
                        else:
                            P.bindings[o] = ("vnf", attr)
                        if attr is not None:
                            P.vnf_bound.append(attr)
                    else:
                        if attr is None:
                            raise UnsupportedRule(f"{name}: constant object of {pl} not supported")
                        P.vnf_consts.append((attr, _const(o)))
                    pending.remove(t)
                    progress = True
                elif s in paths:                # server-side property path
                    path = paths[s] + (pl,)
                    if path in SERVER_PATHS:
                        col = SERVER_PATHS[path]
                        if isinstance(col, SetColumn):
                            if not isinstance(o, Variable):
                                raise UnsupportedRule(f"{name}: constant object of set-valued {pl} not supported")
                            if o in P.bindings and P.bindings[o][0] == "vnf":
                                P.joins.append((col, P.bindings[o][1]))
                                P.validity.append(col)
                            else:
                                P.bindings[o] = ("server", col)
                        elif isinstance(o, Variable):
                            if o in P.bindings and P.bindings[o][0] == "vnf":
                                P.joins.append((col, P.bindings[o][1]))
                            else:
                                P.bindings[o] = ("server", col)
                            if col.validity:
                                P.validity.append(col)
                        else:
                            P.server_consts.append((col, _const(o)))
                            if col.validity:
                                P.validity.append(col)
                    elif path in SERVER_NODES:
                        if not isinstance(o, Variable):
                            raise UnsupportedRule(f"{name}: constant intermediate node on path {path}")
                        paths[o] = path
                    else:
                        raise UnsupportedRule(f"{name}: property path {path} is not bound")
                    pending.remove(t)
                    progress = True
        if pending:
            raise UnsupportedRule(f"{name}: unresolved triple patterns {pending}")
        for expr in filters:
            _check_expr(name, expr, P.bindings)
            P.filters.append(expr)
        return P

    # ------------------------------------------------------------------ code generation
    def _col(self, col) -> str:
        """Reference to a column object in the emitted code."""
        if col not in self.columns:
            self.columns.append(col)
        return f"C[{self.columns.index(col)}]"

    def _root(self, c: str) -> bool:
        """True if every VNF individual belongs to class c (c is a superclass of every VNF class)."""
        return all(c in sup for cls, sup in self.classes.items() if "VNF" in sup)

    def _emit(self, name: str, patterns: list) -> str:
        L = []
        body = [f"def rule_{name}(V, S, deny):"]
        if len(patterns) > 1:
            body.append("    out = np.zeros((V.n, S.N), dtype=bool)")
        for k, P in enumerate(patterns):
            # VNF side of the pattern (cached per chain by V.pattern)
            conds = [f"V.cls({c!r})" for c in P.vnf_classes if not self._root(c)]
            conds += [f"V.has({a!r})" for a in dict.fromkeys(P.vnf_bound)]
            for a, const in P.vnf_consts:
                acc = "V.obj" if isinstance(const, str) else "V.num"
                conds.append(f"V.has({a!r}) & ({acc}({a!r}) == {const!r})")
            L.append(f"def vnf_{name}_{k}(V):")
            L.append("    return " + (" & ".join(conds) if conds else "np.ones(V.n, dtype=bool)"))
            L.append("")
            body.append(f"    # pattern {k}")
            body.append(f"    vnf_ok, nz = V.pattern({name!r}, {k}, vnf_{name}_{k})")
            body.append("    if nz:")
            # server-side conditions that do not depend on the VNF: (N,) vectors, combined before the outer product
            srv = [self._emit_const(col, const) for col, const in P.server_consts]
            pair = []                                   # (n, N) conditions
            for col, a in P.joins:
                if isinstance(col, SetColumn):
                    pair.append(f"S.join({self._col(col)}, V.idx({a!r}, {self._col(col)}.vocabulary))")
                else:
                    pair.append(f"(S.col({self._col(col)})[None, :] == V.obj({a!r})[:, None])")
            for expr in P.filters:
                if _server_only(expr, P.bindings):
                    srv.append(self._emit_expr(expr, P.bindings, vector=True))
                else:
                    pair.append(self._emit_expr(expr, P.bindings))
            if srv:
                body.append("        srv_ok = " + " & ".join(srv))
                body.append("        fire = vnf_ok[:, None] & srv_ok[None, :]")
                for e in pair:
                    body.append(f"        fire &= {e}")
            elif pair:
                body.append(f"        fire = {pair[0]} & vnf_ok[:, None]")
                for e in pair[1:]:
                    body.append(f"        fire &= {e}")
            else:
                body.append("        fire = np.repeat(vnf_ok[:, None], S.N, axis=1)")
            if P.validity:
                unk = []
                for col in dict.fromkeys(P.validity):
                    if isinstance(col, SetColumn):
                        if col.known is not None:
                            unk.append(f"~{self._col(col)}.known(S.kv)")
                    else:
                        unk.append(f"~S.col({self._col(col)})")
                if unk:
                    body.append("        if deny:")
                    body.append("            fire |= vnf_ok[:, None] & (" + " | ".join(unk) + ")[None, :]")
            if len(patterns) > 1:
                body.append("        out |= fire")
            else:
                body.append("        return fire")
                body.append("    return np.zeros((V.n, S.N), dtype=bool)")
        if len(patterns) > 1:
            body.append("    return out")
        return "\n".join(L + body) + "\n"

    def _emit_const(self, col, const) -> str:
        """Equality of a column with a constant (individuals of boolean-coded columns become flags)."""
        if isinstance(const, str) and col.codes:
            code = dict(col.codes).get(const)
            if code is None:
                raise UnsupportedRule(f"{col.name}: individual {const} is not in the coded vocabulary")
            const = code
        if col.boolean and isinstance(const, bool):
            return f"S.col({self._col(col)})" if const else f"~S.col({self._col(col)})"
        return f"(S.col({self._col(col)}) == {const!r})"

    def _emit_expr(self, expr, bindings, vector: bool = False) -> str:
        """Python/NumPy source of a FILTER expression: (n, N) by broadcasting, or (N,) if ``vector``."""
        if isinstance(expr, Variable):
            side, ref = bindings[expr]
            if side == "server":
                return f"S.col({self._col(ref)})" if vector else f"S.col({self._col(ref)})[None, :]"
            if ref is None:
                raise UnsupportedRule(f"variable {expr} has no value")
            return f"V.num({ref!r})[:, None]"
        if isinstance(expr, (Literal, URIRef)):
            return repr(_const(expr))
        if expr.name == "RelationalExpression":
            op = {"=": "==", "!=": "!="}.get(expr.op, expr.op)
            return f"({self._emit_expr(expr.expr, bindings, vector)} {op} {self._emit_expr(expr.other, bindings, vector)})"
        if expr.name == "ConditionalAndExpression":
            return "(" + " & ".join(self._emit_expr(e, bindings, vector) for e in [expr.expr] + list(expr.other)) + ")"
        if expr.name == "ConditionalOrExpression":
            return "(" + " | ".join(self._emit_expr(e, bindings, vector) for e in [expr.expr] + list(expr.other)) + ")"
        if expr.name == "UnaryNot":
            return f"(~{self._emit_expr(expr.expr, bindings, vector)})"
        raise UnsupportedRule(f"unsupported expression {expr}")

    # ------------------------------------------------------------------ evaluation
    def _closure(self, v):
        key = v.classes if isinstance(v.classes, frozenset) else frozenset(v.classes)
        cl = self._closure_cache.get(key)
        if cl is None:
            cl = {"VNF"} | set().union(*(self.classes.get(c, {c}) for c in key)) | set(self.classes.get("VNF", ()))
            self._closure_cache[key] = cl
        return cl

    def evaluate(self, vnfs, kv, enabled, unknown: str = "permit") -> dict:
        V, S = _VnfView(vnfs, self._closure, self._vcache), _SrvView(kv)
        deny = unknown == "deny"
        return {name: fn(V, S, deny) for name, fn in self.functions.items() if name in enabled}


def _server_only(expr, bindings) -> bool:
    """True if the FILTER expression references no VNF-side variable (it is an (N,) condition)."""
    if isinstance(expr, Variable):
        return bindings[expr][0] == "server"
    if isinstance(expr, (Literal, URIRef)):
        return True
    if expr.name == "RelationalExpression":
        return _server_only(expr.expr, bindings) and _server_only(expr.other, bindings)
    if expr.name in ("ConditionalAndExpression", "ConditionalOrExpression"):
        return all(_server_only(e, bindings) for e in [expr.expr] + list(expr.other))
    if expr.name == "UnaryNot":
        return _server_only(expr.expr, bindings)
    return False


def _check_expr(name, expr, bindings):
    if isinstance(expr, Variable):
        if expr not in bindings:
            raise UnsupportedRule(f"{name}: unbound variable {expr} in FILTER")
        return
    if isinstance(expr, (Literal, URIRef)):
        return
    if isinstance(expr, CompValue):
        if expr.name == "RelationalExpression":
            if expr.op not in ("<", "<=", "=", "!=", ">=", ">"):
                raise UnsupportedRule(f"{name}: operator {expr.op} not supported")
            _check_expr(name, expr.expr, bindings)
            _check_expr(name, expr.other, bindings)
            return
        if expr.name in ("ConditionalAndExpression", "ConditionalOrExpression"):
            _check_expr(name, expr.expr, bindings)
            for e in expr.other:
                _check_expr(name, e, bindings)
            return
        if expr.name == "UnaryNot":
            _check_expr(name, expr.expr, bindings)
            return
    raise UnsupportedRule(f"{name}: unsupported FILTER expression {expr}")


_PROGRAM = None


def program() -> Program:
    """The compiled program of ``ontology/policies.rq`` (compiled once per process)."""
    global _PROGRAM
    if _PROGRAM is None:
        _PROGRAM = Program(RULE_TEXTS)
    return _PROGRAM


def describe(prog: Program | None = None) -> str:
    """The emitted source of all rules (for inspection; e.g. ``python -m semvnf.compiler``)."""
    prog = prog or program()
    return "\n".join(prog.source.values())


if __name__ == "__main__":
    print(describe())
