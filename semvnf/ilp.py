"""Semantic-ILP: exact arc-flow formulation of the single-request problem (Section 3.4).

    min  sum_ij c_ij x_ij + sum_j c_j y_j + sum_e sum_a c_a f^e_a                    (24)
    s.t. sum_j x_ij = 1                                   for all i                  (25)
         sum_i r_ik x_ij <= c_jk                          for all j, k               (26)
         sum_e b f^e_a <= c_a                             for all arcs a             (27)
         sum_out f^e - sum_in f^e = src^e_u - dst^e_u     for all e, u               (28)
         x_ij <= y_j                                      for idle servers j         (29)
         sum_ij d_ij x_ij + sum_e sum_a l_a f^e_a <= L_max                           (30)
         x_ij = 0                                         for (j, i) in I_sem        (31)
         x, y, f binary                                                              (32)

The semantic constraint (31) is applied by not creating the pruned variables (variable
fixing before the solver is called).  Solved with HiGHS (open source, via SciPy) or with
Gurobi when the model fits the licence.  Cycles in the returned flows (possible only
when the solver stops before optimality) are removed by ``remove_cycles``.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .model import Instance, remove_cycles


def build(inst: Instance, use_pruning: bool = True):
    """Return the sparse MILP (c, A, lb, ub, var maps)."""
    n, N = inst.n, inst.N
    r = inst.req
    dom = inst.cand if use_pruning else inst.cap_ok           # without pruning: policies become rows
    xs = [(i, j) for i in range(n) for j in np.flatnonzero(dom[i])]
    xidx = {p: k for k, p in enumerate(xs)}
    servers = sorted({j for _, j in xs})
    ys = [j for j in servers if not inst.active[j]]
    yidx = {j: len(xs) + k for k, j in enumerate(ys)}
    arcs_ok = np.flatnonzero(inst.usable & (inst.res_bw >= r.bw - 1e-9))
    n_e = n + 1
    base_f = len(xs) + len(ys)
    fidx = {}
    for e in range(n_e):
        for k, a in enumerate(arcs_ok):
            fidx[(e, int(a))] = base_f + e * len(arcs_ok) + k
    base_z = base_f + n_e * len(arcs_ok)
    zs = servers if inst.c_h is not None else []
    zidx = {j: base_z + k for k, j in enumerate(zs)}
    nv = base_z + len(zs)
    c = np.zeros(nv)
    integrality = np.ones(nv)
    ubv = np.ones(nv)
    for j, k in zidx.items():
        c[k] = inst.c_h_vec[j]
        integrality[k] = 0
        ubv[k] = np.inf
    for (i, j), k in xidx.items():
        c[k] = inst.c_x[i, j]
    for j, k in yidx.items():
        c[k] = inst.c_y[j]
    for (e, a), k in fidx.items():
        c[k] = inst.c_f[a]
    rows, cols, vals, lb, ub = [], [], [], [], []
    row = 0

    def add(coefs, lo, hi):
        nonlocal row
        for k, v in coefs:
            rows.append(row)
            cols.append(k)
            vals.append(v)
        lb.append(lo)
        ub.append(hi)
        row += 1

    # (25) assignment
    for i in range(n):
        add([(xidx[(i, j)], 1.0) for j in np.flatnonzero(dom[i])], 1.0, 1.0)
    # (26) node capacity (cpu, mem)
    for j in servers:
        cx = [(xidx[(i, j)], inst.cpu[i]) for i in range(n) if (i, j) in xidx]
        if sum(v for _, v in cx) > inst.res_cpu[j] + 1e-9:
            add(cx, -np.inf, inst.res_cpu[j])
        mx = [(xidx[(i, j)], inst.mem[i]) for i in range(n) if (i, j) in xidx]
        if sum(v for _, v in mx) > inst.res_mem[j] + 1e-9:
            add(mx, -np.inf, inst.res_mem[j])
    # (27) link capacity, only where several virtual links could exceed the residual bandwidth
    for a in arcs_ok:
        if n_e * r.bw > inst.res_bw[a] + 1e-9:
            add([(fidx[(e, int(a))], r.bw) for e in range(n_e)], -np.inf, inst.res_bw[a])
    # (28) flow conservation with ingress / egress endpoints
    out_arcs = {u: [] for u in range(N)}
    in_arcs = {u: [] for u in range(N)}
    for a in arcs_ok:
        u, v = inst.sub.arcs[a]
        out_arcs[int(u)].append(int(a))
        in_arcs[int(v)].append(int(a))
    for e in range(n_e):
        for u in range(N):
            coefs = [(fidx[(e, a)], 1.0) for a in out_arcs[u]] + [(fidx[(e, a)], -1.0) for a in in_arcs[u]]
            rhs = 0.0
            if e == 0:
                rhs += 1.0 if u == r.ingress else 0.0
            elif (e - 1, u) in xidx:
                coefs.append((xidx[(e - 1, u)], -1.0))
            if e == n:
                rhs -= 1.0 if u == r.egress else 0.0
            elif (e, u) in xidx:
                coefs.append((xidx[(e, u)], 1.0))
            if not coefs:
                if abs(rhs) > 0:
                    add([], rhs, rhs)          # infeasible marker (isolated endpoint)
                continue
            add(coefs, rhs, rhs)
    # (29) activation of idle servers
    for (i, j), k in xidx.items():
        if j in yidx:
            add([(k, 1.0), (yidx[j], -1.0)], -np.inf, 0.0)
    # headroom-preserving cost: z_j >= sum_i (r_i/C_j) x_ij - max(0, theta - u_j)
    for j, kz in zidx.items():
        coefs = [(xidx[(i, j)], inst.cpu[i] / inst.sub.cpu_cap[j]) for i in range(n) if (i, j) in xidx]
        coefs.append((kz, -1.0))
        u0 = inst.used_cpu[j] / inst.sub.cpu_cap[j]
        add(coefs, -np.inf, max(0.0, inst.theta_h - u0))
    # (30) end-to-end latency SLA
    coefs = [(k, inst.d_proc[i, j]) for (i, j), k in xidx.items()]
    coefs += [(k, inst.arc_lat[a]) for (e, a), k in fidx.items()]
    add(coefs, -np.inf, r.sla_ms)
    # policies as explicit rows when pruning is disabled (same feasible set, larger model)
    n_policy_rows = 0
    if not use_pruning:
        for (i, j), k in xidx.items():
            if not inst.allowed[i, j]:
                add([(k, 1.0)], 0.0, 0.0)
                n_policy_rows += 1
    A = coo_matrix((vals, (rows, cols)), shape=(row, nv)).tocsr()
    return dict(c=c, A=A, lb=np.array(lb), ub=np.array(ub), nv=nv, integrality=integrality, ubv=ubv,
                zidx=zidx, xs=xs, xidx=xidx, ys=ys, yidx=yidx,
                fidx=fidx, arcs_ok=arcs_ok, n_x=len(xs), n_y=len(ys), n_f=len(fidx), n_rows=row,
                n_policy_rows=n_policy_rows)


def _decode(inst: Instance, M, sol):
    n = inst.n
    servers = [None] * n
    for (i, j), k in M["xidx"].items():
        if sol[k] > 0.5:
            servers[i] = int(j)
    if any(s is None for s in servers):
        return None, None, 0
    ends = inst.path_endpoints(servers)
    paths, removed = [], 0
    for e in range(n + 1):
        used = [a for (ee, a), k in M["fidx"].items() if ee == e and sol[k] > 0.5]
        p = remove_cycles(used, inst.sub.arcs, ends[e][0], ends[e][1])
        if p is None:
            return None, None, 0
        removed += len(used) - len(p)
        paths.append(p)
    return servers, paths, removed


def solve(inst: Instance, time_limit: float = 60.0, mip_gap: float = 1e-4, use_pruning: bool = True,
          solver: str = "highs") -> dict:
    """Solve the single-request ILP; returns servers, paths, status, model size and timings."""
    t0 = time.perf_counter()
    if not use_pruning or inst.cand.any(axis=1).all():
        M = build(inst, use_pruning=use_pruning)
    else:
        return dict(servers=None, paths=None, status="infeasible_domain", t_build=time.perf_counter() - t0,
                    t_solve=0.0, n_x=0, n_y=0, n_f=0, n_rows=0, obj=None, gap=None, cycles_removed=0)
    t1 = time.perf_counter()
    if solver == "gurobi":
        try:
            res = _solve_gurobi(M, time_limit, mip_gap)
        except Exception as e:                      # e.g. size-limited licence
            res = dict(x=None, status="gurobi_error", obj=None, gap=None, error=str(e)[:80])
    else:
        cons = LinearConstraint(M["A"], M["lb"], M["ub"])
        r = milp(M["c"], constraints=cons, integrality=M["integrality"], bounds=Bounds(np.zeros(M["nv"]), M["ubv"]),
                 options=dict(time_limit=time_limit, mip_rel_gap=mip_gap, disp=False))
        status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "other"}.get(r.status, "other")
        if r.status == 1 and r.x is None:
            status = "time_limit_nosol"
        res = dict(x=r.x, status=status, obj=(float(r.fun) if r.x is not None else None),
                   gap=getattr(r, "mip_gap", None))
    t2 = time.perf_counter()
    servers = paths = None
    removed = 0
    if res["x"] is not None:
        servers, paths, removed = _decode(inst, M, res["x"])
    return dict(servers=servers, paths=paths, status=res["status"], obj=res["obj"], gap=res.get("gap"),
                t_build=t1 - t0, t_solve=t2 - t1, n_x=M["n_x"], n_y=M["n_y"], n_f=M["n_f"],
                n_rows=M["n_rows"], cycles_removed=removed)


def _solve_gurobi(M, time_limit, mip_gap):
    import gurobipy as gp
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = 1
    vt = np.where(M["integrality"] > 0, gp.GRB.BINARY, gp.GRB.CONTINUOUS)
    x = m.addMVar(M["nv"], lb=0.0, ub=M["ubv"], vtype=vt)
    m.setObjective(M["c"] @ x, gp.GRB.MINIMIZE)
    A, lb, ub = M["A"], M["lb"], M["ub"]
    eq = np.isclose(lb, ub)
    if eq.any():
        m.addConstr(A[eq] @ x == lb[eq])
    le = ~eq & np.isfinite(ub)
    if le.any():
        m.addConstr(A[le] @ x <= ub[le])
    ge = ~eq & np.isfinite(lb)
    if ge.any():
        m.addConstr(A[ge] @ x >= lb[ge])
    m.optimize()
    st = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit", gp.GRB.INFEASIBLE: "infeasible"}.get(m.Status, "other")
    if m.SolCount > 0:
        return dict(x=x.X, status=st, obj=float(m.ObjVal), gap=float(m.MIPGap))
    return dict(x=None, status=st, obj=None, gap=None)
