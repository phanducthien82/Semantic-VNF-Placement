"""Analysis of experiments E2-E9 and auxiliary facts for make_report.py."""
from __future__ import annotations

import collections
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, ROOT, load_runs

from semvnf.stats import mean_ci, paired

TOPOS = ("euro28", "germany50")
TOPO_LABEL = {"euro28": "Euro28", "germany50": "Germany50"}
ALGS = ("Semantic-ILP", "Semantic-GA", "GA", "Greedy", "Random", "DDPG", "GCN-PPO")
COL = {"Semantic-ILP": "#0072B2", "Semantic-GA": "#D55E00", "GA": "#009E73", "Greedy": "#CC79A7",
       "Random": "#999999", "DDPG": "#E69F00", "GCN-PPO": "#56B4E9"}
MRK = {"Semantic-ILP": "o", "Semantic-GA": "s", "GA": "^", "Greedy": "D", "Random": "x", "DDPG": "v",
       "GCN-PPO": "P"}


def pstars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "$^{\\mathrm{ns}}$"


# ====================================================================== workload / repo facts
def facts(R):
    from semvnf.catalog import mean_chain_length, mean_cpu_per_sfc
    R.m("wl/cpu_per_sfc", mean_cpu_per_sfc(), "{:.1f}")
    R.m("wl/chain_len", mean_chain_length(), "{:.2f}")
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        c = "n/a"
    R.m("repo/commit", c)


# ====================================================================== DRL timing (R2.12)
def drl_timing(R):
    """Training and per-decision cost of every method.  Per-decision times are taken from E1-T
    (identical low-load conditions) when available, otherwise from the E1 runs."""
    rows = load_runs("e1_timing")
    src = "e1t"
    if not rows:
        rows, src = load_runs("e1_online"), "e1"
    R.m("drl/source", src)
    R.m("drl/seeds", len({r["job"]["seed"] for r in rows}), "{:d}")
    lines = []
    for t in TOPOS:
        for a in ("Semantic-ILP", "Semantic-GA", "GA", "Greedy", "DDPG", "GCN-PPO"):
            rs = [r for r in rows if r["job"]["topo"] == t and r["job"]["alg"] == a]
            if not rs:
                continue
            train = None
            drl = a in ("DDPG", "GCN-PPO")
            if drl:
                pth = RESULTS / "drl" / f"{t}_{a}.json"
                if pth.exists():
                    d = json.loads(pth.read_text())
                    train = d["train_time_s"]
                    R.m(f"drl/{t}/{a}/train_s", train, "{:.0f}")
                    R.m(f"drl/{t}/{a}/train_min", train / 60, "{:.1f}")
                    R.m(f"drl/{t}/{a}/traces", d["n_traces"], "{:d}")
            t_load = float(np.mean([r.get("t_load_s") or 0.0 for r in rs])) * 1e3
            t_sem = float(np.mean([r["t_sem_ms"] for r in rs]))
            t_pre = float(np.mean([r["t_pre_ms"] for r in rs]))
            t_inf = float(np.mean([r["t_inf_ms"] for r in rs]))
            t_alg = float(np.mean([r["t_alg_ms"] for r in rs]))
            t_route = max(0.0, t_alg - t_pre - t_inf) if drl else None
            t_dec = float(np.mean([r["t_dec_ms"] for r in rs]))
            p95 = float(np.mean([r["t_dec_p95_ms"] for r in rs]))
            for k, v in (("t_load_ms", t_load), ("t_sem_ms", t_sem), ("t_pre_ms", t_pre), ("t_inf_ms", t_inf),
                         ("t_dec_ms", t_dec), ("t_p95_ms", p95), ("t_alg_ms", t_alg)):
                R.m(f"drl/{t}/{a}/{k}", v, "{:.3f}" if v < 1 else ("{:.2f}" if v < 100 else "{:.0f}"))
            if t_route is not None:
                R.m(f"drl/{t}/{a}/t_route_ms", t_route, "{:.3f}" if t_route < 1 else "{:.2f}")
            f = lambda v: "--" if v is None else (f"{v:.3f}" if v < 1 else (f"{v:.2f}" if v < 100 else f"{v:.0f}"))
            lines.append(f"{TOPO_LABEL[t]} & {a} & {('%.1f' % (train / 60)) if train else '--'} & "
                         f"{f(t_load) if drl else '--'} & {f(t_sem)} & {f(t_pre) if drl else '--'} & "
                         f"{f(t_inf) if drl else '--'} & {f(t_route)} & {'--' if drl else f(t_alg)} & "
                         f"{f(t_dec)} & {f(p95)} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    # range of the DRL decision times over agents and topologies, and their share of inference
    drl_dec = [float(np.mean([r["t_dec_ms"] for r in rows if r["job"]["topo"] == t and r["job"]["alg"] == a]))
               for t in TOPOS for a in ("DDPG", "GCN-PPO")
               if any(r["job"]["topo"] == t and r["job"]["alg"] == a for r in rows)]
    if drl_dec:
        R.m("drl/dec_min", min(drl_dec), "{:.2f}")
        R.m("drl/dec_max", max(drl_dec), "{:.2f}")
    # ratios used in the text (decision time of the exact model relative to the fastest learned policy)
    for t in TOPOS:
        try:
            ilp = float(np.mean([r["t_dec_ms"] for r in rows if r["job"]["topo"] == t and r["job"]["alg"] == "Semantic-ILP"]))
            sga = float(np.mean([r["t_dec_ms"] for r in rows if r["job"]["topo"] == t and r["job"]["alg"] == "Semantic-GA"]))
            gre = float(np.mean([r["t_dec_ms"] for r in rows if r["job"]["topo"] == t and r["job"]["alg"] == "Greedy"]))
            R.m(f"drl/{t}/sga_over_ilp", sga / ilp, "{:.1f}")
            R.m(f"drl/{t}/ilp_over_greedy", ilp / gre, "{:.0f}")
        except (ZeroDivisionError, ValueError):
            pass
    if lines:
        body = ("\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n"
                "Topology & Method & Training (min) & Load (ms) & Rules (ms) & State (ms) & Inference (ms) & "
                "Routing (ms) & Solver (ms) & Decision (ms) & p95 (ms) \\\\\n\\midrule\n" + "\n".join(lines)
                + "\n\\bottomrule\n\\end{tabular}\n")
        (R.tex / "tables" / "tab_drl_timing.tex").write_text(body, encoding="utf-8")


# ====================================================================== trade-off figure (R1.13)
def fig_tradeoff(R):
    rows = load_runs("e1_online")
    if not rows:
        return
    g = R.by(rows, "topo", "alg")
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.0))
    for col, t in enumerate(TOPOS):
        ax1, ax2 = axes[2 * col], axes[2 * col + 1]
        for a in ALGS:
            if (t, a) not in g:
                continue
            rs = g[(t, a)]
            kw = dict(color=COL[a], marker=MRK[a], s=24, alpha=0.85, linewidths=0.5, label=a)
            if MRK[a] != "x":
                kw["edgecolors"] = "black"
            ax1.scatter([r["t_dec_ms"] for r in rs], [r["energy_per_acc_wh"] for r in rs], **kw)
            ax2.scatter([r["latency"] for r in rs], [100 * r["acceptance"] for r in rs], **kw)
        ax1.set_xscale("log")
        ax1.set_xlabel("Decision time (ms, log)")
        ax1.set_ylabel("Energy per accepted SFC (Wh)")
        ax1.set_title(f"{TOPO_LABEL[t]}: energy vs. decision time")
        ax2.set_xlabel("Mean end-to-end latency (ms)")
        ax2.set_ylabel("Acceptance ratio (%)")
        ax2.set_title(f"{TOPO_LABEL[t]}: acceptance vs. latency")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    R.save(fig, "fig_tradeoff")


# ====================================================================== E2/E3/E6/E10 shadow evaluation
def _summ(xs):
    feas = [x for x in xs if x["ilp_ok"]]
    solved = [x for x in feas if x["ok"]]
    gaps = [x["gap"] for x in solved if x["gap"] is not None]
    infeas = [x for x in xs if not x["ilp_ok"]]
    return dict(n=len(xs), n_feas=len(feas), solved=100 * len(solved) / max(len(feas), 1),
                gap_mean=100 * float(np.mean(gaps)) if gaps else np.nan,
                gap_med=100 * float(np.median(gaps)) if gaps else np.nan,
                gap_p95=100 * float(np.percentile(gaps, 95)) if gaps else np.nan,
                opt=100 * float(np.mean([g < 1e-6 for g in gaps])) if gaps else np.nan,
                t_ms=1e3 * float(np.mean([x["t"] for x in xs])),
                false_ok=100 * float(np.mean([x["ok"] for x in infeas])) if infeas else 0.0)


def e2(R):
    rows = load_runs("e2_shadow")
    if not rows:
        return
    recs = []
    for r in rows:
        for x in r["records"]:
            x = dict(x)
            x["topo"] = r["job"]["topo"]
            x["seed"] = r["job"]["seed"]
            recs.append(x)
    by = collections.defaultdict(list)
    pooled = collections.defaultdict(list)
    for x in recs:
        by[(x["topo"], x["solver"])].append(x)
        pooled[x["solver"]].append(x)
    stats = {k: _summ(v) for k, v in by.items()}
    fm = lambda k: "{:d}" if k in ("n", "n_feas") else ("{:.2f}" if k.startswith("gap") else "{:.1f}")
    for (t, sv), st in stats.items():
        for k, v in st.items():
            R.m(f"e2/{t}/{sv}/{k}", v, fm(k))
    for sv, xs in pooled.items():
        for k, v in _summ(xs).items():
            R.m(f"e2/all/{sv}/{k}", v, fm(k))
    R.m("e2/snapshots", len(pooled.get("Semantic-GA", [])), "{:d}")
    R.m("e2/snapshots_feas", sum(1 for x in pooled.get("Semantic-GA", []) if x["ilp_ok"]), "{:d}")
    R.m("e2/seeds", len(rows) // 2, "{:d}")
    # ---- GA ablation (E3): per-seed mean gap, paired Wilcoxon vs Semantic-GA over seeds
    variants = ["GA", "GA+P", "GA+W", "GA+P+W", "Semantic-GA"]
    seedstat = collections.defaultdict(dict)
    for (t, sv), xs in by.items():
        for sd in {x["seed"] for x in xs}:
            sx = [x for x in xs if x["seed"] == sd and x["ilp_ok"]]
            if not sx:
                continue
            gaps = [x["gap"] for x in sx if x["ok"] and x["gap"] is not None]
            seedstat[(t, sv)][sd] = (100 * float(np.mean(gaps)) if gaps else np.nan, 100 * float(np.mean([x["ok"] for x in sx])))
    names = {"GA": "GA (pure)", "GA+P": "GA + pruning", "GA+W": "GA + warm start",
             "GA+P+W": "GA + pruning + warm start", "Semantic-GA": "\\textbf{Semantic-GA} (+ local search)"}
    lines = []
    for t in TOPOS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{TOPO_LABEL[t]}}} ({stats.get((t, 'Semantic-GA'), {}).get('n_feas', 0)} feasible snapshots)}} \\\\")
        for sv in variants:
            st = stats.get((t, sv))
            if not st:
                continue
            ptxt = "--"
            if sv != "Semantic-GA" and (t, "Semantic-GA") in seedstat:
                a, b = seedstat[(t, "Semantic-GA")], seedstat[(t, sv)]
                sds = sorted(set(a) & set(b))
                pr = paired([a[s][0] for s in sds], [b[s][0] for s in sds])
                ptxt = f"{pr['p_wilcoxon']:.3f}" if pr["p_wilcoxon"] >= 0.001 else "$<$0.001"
                R.m(f"e3/{t}/{sv}/p", pr["p_wilcoxon"], "{:.3f}")
            lines.append(f"{names[sv]} & {st['solved']:.1f} & {st['gap_mean']:.2f} & {st['gap_p95']:.2f} & "
                         f"{st['opt']:.1f} & {st['t_ms']:.1f} & {ptxt} \\\\")
    body = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
            "Variant (5000 evaluations each) & Solved (\\%) & Mean gap (\\%) & p95 gap (\\%) & Optimal (\\%) & Time (ms) & $p$ (vs.\\ S-GA) \\\\\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (R.tex / "tables" / "tab_e3_ablation.tex").write_text(body, encoding="utf-8")
    # ---- path-set size K and Yen (E10)
    ks = [("SGA-K1", 1), ("SGA-K2", 2), ("SGA-K3", 3), ("Semantic-GA", 5), ("SGA-K8", 8)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))
    for t in TOPOS:
        pts = [(k, stats[(t, sv)]) for sv, k in ks if (t, sv) in stats]
        if not pts:
            continue
        mk = "o" if t == "euro28" else "s"
        axes[0].plot([k for k, _ in pts], [s["gap_mean"] for _, s in pts], marker=mk, label=TOPO_LABEL[t])
        axes[1].plot([k for k, _ in pts], [s["solved"] for _, s in pts], marker=mk, label=TOPO_LABEL[t])
        axes[2].plot([k for k, _ in pts], [s["t_ms"] for _, s in pts], marker=mk, label=TOPO_LABEL[t])
        y = stats.get((t, "SGA-Yen5"))
        if y:
            for ax, key in zip(axes, ("gap_mean", "solved", "t_ms")):
                ax.scatter([5], [y[key]], marker="*", s=80, color="black", zorder=5,
                           label="Yen K=5" if (t == "euro28" and key == "gap_mean") else None)
    axes[0].set_ylabel("Mean optimality gap (%)")
    axes[1].set_ylabel("Solved when S-ILP feasible (%)")
    axes[2].set_ylabel("Decision time (ms)")
    for ax in axes:
        ax.set_xlabel("Path-set size $K$")
        ax.set_xticks([1, 2, 3, 5, 8])
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    R.save(fig, "fig_e10_k")
    for sv, k in ks + [("SGA-Yen5", "yen")]:
        for t in TOPOS:
            st = stats.get((t, sv))
            if st:
                R.m(f"e10/{t}/{k}/gap", st["gap_mean"], "{:.2f}")
                R.m(f"e10/{t}/{k}/solved", st["solved"])
                R.m(f"e10/{t}/{k}/t", st["t_ms"])
    # ---- penalty sensitivity (E6)
    oms = [("GA-static-0.01", "0.01"), ("GA-static-0.1", "0.1"), ("GA-static-1", "1"), ("GA-static-10", "10"),
           ("GA-static-100", "100"), ("GA-segregated", "ff")]
    lines = []
    for sv, lab in oms:
        xs = pooled.get(sv)
        if not xs:
            continue
        st = _summ(xs)
        feas = [x for x in xs if x["ilp_ok"]]
        inf_ret = 100 * float(np.mean([not x["ok"] for x in feas])) if feas else np.nan
        R.m(f"e6/{lab}/infeasible_returned", inf_ret)
        R.m(f"e6/{lab}/gap", st["gap_mean"], "{:.2f}")
        name = f"Static, $\\Omega={lab}$" if lab != "ff" else "Feasibility-first, Eq.~\\eqref{eq:fitness}"
        lines.append(f"{name} & {inf_ret:.1f} & {st['gap_mean']:.2f} & {st['t_ms']:.1f} \\\\")
    if lines:
        body = ("\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nPenalty (GA without pruning) & Infeasible returned (\\%) & Mean gap (\\%) & Time (ms) \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
        (R.tex / "tables" / "tab_e6_penalty.tex").write_text(body, encoding="utf-8")
    # ---- MILP with/without pruning; Gurobi cross-check; measured search-space reduction
    for t in TOPOS:
        a = by.get((t, "ILP-nopruning"), [])
        if a:
            R.m(f"e2/{t}/nopr/nx", np.mean([x["n_x"] for x in a]), "{:.0f}")
            R.m(f"e2/{t}/nopr/rows", np.mean([x["n_rows"] for x in a]), "{:.0f}")
            R.m(f"e2/{t}/nopr/t_ms", 1e3 * np.mean([x["t"] for x in a]), "{:.1f}")
            same = [abs(x["gap"]) < 1e-6 for x in a if x["gap"] is not None]
            R.m(f"e2/{t}/nopr/same", 100 * np.mean(same) if same else None)
        gu = by.get((t, "ILP-gurobi"), [])
        if gu and a:
            # model sizes on the epochs where both models were built and the problem is feasible
            key = lambda x: (x["seed"], x["sfc"])
            pa = {key(x): x for x in a if x["ilp_ok"] and x["n_x"] > 0}
            pg = {key(x): x for x in gu if x["ilp_ok"] and x["n_x"] > 0}
            both = sorted(set(pa) & set(pg))
            if both:
                for lab, src in (("pruned", pg), ("nopr", pa)):
                    R.m(f"e2/{t}/{lab}/nx", np.mean([src[k]["n_x"] for k in both]), "{:.0f}")
                    R.m(f"e2/{t}/{lab}/nf", np.mean([src[k]["n_f"] for k in both]), "{:.0f}")
                    R.m(f"e2/{t}/{lab}/rows", np.mean([src[k]["n_rows"] for k in both]), "{:.0f}")
            R.m(f"e2/{t}/gurobi/t_ms", 1e3 * np.mean([x["t"] for x in gu]), "{:.1f}")
        if gu:
            ok = [x for x in gu if x.get("status") != "gurobi_error"]
            R.m(f"e2/{t}/gurobi/n", len(ok), "{:d}")
            R.m(f"e2/{t}/gurobi/share", 100 * len(ok) / len(gu))
            diffs = [abs(x["gap"]) for x in ok if x["gap"] is not None]
            R.m(f"e2/{t}/gurobi/maxdiff", 100 * max(diffs) if diffs else 0.0, "{:.4f}")
        xs = by.get((t, "Semantic-GA"), [])
        if xs:
            red = [x["phys_log10"] - x["domain_log10"] for x in xs]
            R.m(f"e2/{t}/log10R/mean", float(np.mean(red)), "{:.2f}")
            R.m(f"e2/{t}/log10R/max", float(np.max(red)), "{:.2f}")
            per_tpl = collections.defaultdict(list)
            for x in xs:
                per_tpl[x["tpl"]].append(x["phys_log10"] - x["domain_log10"])
            for tpl, v in per_tpl.items():
                R.m(f"e2/{t}/log10R/{tpl}", float(np.mean(v)), "{:.2f}")
        ilp = [x for x in by.get((t, "ILP-nopruning"), [])]
        if ilp and xs:
            pass
    # ---- convergence figure (E3)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    grid = np.arange(0, 5001, 100)
    colr = {"GA": "#009E73", "GA+P": "#0072B2", "GA+W": "#CC79A7", "GA+P+W": "#E69F00", "Semantic-GA": "#D55E00"}
    sty = {"GA": "-", "GA+P": "--", "GA+W": "-.", "GA+P+W": ":", "Semantic-GA": "-"}
    for sv in variants:
        cf, cg = [], []
        for x in pooled.get(sv, []):
            if not x.get("hist") or not x["ilp_ok"] or x["ilp_J"] is None:
                continue
            h = np.array(x["hist"], dtype=float)
            ev, J, V = h[:, 0], h[:, 2], h[:, 3]
            idx = np.searchsorted(ev, grid, side="right") - 1
            ok = idx >= 0
            ii = np.clip(idx, 0, None)
            feas = ok & (V[ii] <= 1e-12)
            gap = np.where(feas, (J[ii] - x["ilp_J"]) / max(abs(x["ilp_J"]), 1e-9), np.nan)
            cf.append(feas.astype(float))
            cg.append(gap)
        if not cf:
            continue
        axes[0].plot(grid, 100 * np.mean(cf, axis=0), sty[sv], color=colr[sv], label=sv, lw=1.4)
        with np.errstate(all="ignore"):
            med = 100 * np.nanmedian(np.array(cg), axis=0)
        axes[1].plot(grid, med, sty[sv], color=colr[sv], label=sv, lw=1.4)
    axes[0].set_ylabel("Snapshots with feasible incumbent (%)")
    axes[1].set_ylabel("Median gap of incumbent (%)")
    axes[1].set_yscale("symlog", linthresh=0.1)
    for ax in axes:
        ax.set_xlabel("Fitness evaluations")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    R.save(fig, "fig_e3_convergence")


# ====================================================================== E4 policy ablation
def _e4b(R):
    """How often each policy type fires on the Semantic-ILP trajectory with all policies (E4b)."""
    rows = load_runs("e4b_policy_shares")
    out = {}
    from semvnf.rules import POLICIES
    for t in TOPOS:
        rs = [r for r in rows if r["job"]["topo"] == t]
        if not rs:
            continue
        for p in POLICIES + ("any",):
            d = {k: 100 * float(np.mean([r[f"{p}/{k}"] for r in rs])) for k in ("pairs", "unique", "blocks")}
            out[(t, p)] = d
            for k, v in d.items():
                R.m(f"e4b/{t}/{p}/{k}", v)
        R.m(f"e4b/{t}/seeds", len(rs), "{:d}")
    return out


def e4(R):
    rows = load_runs("e4_policies")
    if not rows:
        return
    from semvnf.rules import POLICIES
    shares = _e4b(R)
    g = R.by(rows, "topo", "alg", "cfg")
    keys = ("acceptance", "energy_per_acc_wh", "carbon_per_acc_g", "latency", "mc_exposure", "pruned_frac",
            "space_log10", "full_log10", "t_dec_ms", "t_sem_ms", "t_alg_ms")
    res = {}
    for (t, a, c), rs in g.items():
        d = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        d["policy_rej"] = float(np.mean([r["reasons"].get("policy", 0) / max(r["arrivals"], 1) for r in rs]))
        d["seeds"] = {r["job"]["seed"]: r for r in rs}
        res[(t, a, c)] = d

    def cells(x, base):
        d_acc = 100 * (x["acceptance"] - base["acceptance"])
        d_en = 100 * (x["energy_per_acc_wh"] - base["energy_per_acc_wh"]) / base["energy_per_acc_wh"]
        d_co = 100 * (x["carbon_per_acc_g"] - base["carbon_per_acc_g"]) / base["carbon_per_acc_g"]
        d_lat = 100 * (x["latency"] - base["latency"]) / base["latency"]
        d_mc = 100 * (x["mc_exposure"] - base["mc_exposure"])
        d_sp = x["space_log10"] - base["space_log10"]
        return d_acc, d_en, d_co, d_lat, d_mc, d_sp

    def sh(t, p):
        d = shares.get((t, p))
        return ("--", "--", "--") if d is None else (f"{d['pairs']:.1f}", f"{d['unique']:.1f}", f"{d['blocks']:.1f}")

    lines, rank = [], collections.defaultdict(list)
    for t in TOPOS:
        base = res.get((t, "Semantic-ILP", "all"))
        if not base:
            continue
        n_seeds = len(base["seeds"])
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{TOPO_LABEL[t]}}}: all policies -- acceptance "
                     f"{100 * base['acceptance']:.1f}\\%, energy/acc.\\ {base['energy_per_acc_wh']:.2f}~Wh, "
                     f"CO$_2$/acc.\\ {base['carbon_per_acc_g']:.2f}~g, latency {base['latency']:.1f}~ms, "
                     f"MC exposure {100 * base['mc_exposure']:.1f}\\%, "
                     f"$\\log_{{10}}\\prod_i|\\Cset_i|={base['space_log10']:.2f}$}} \\\\")
        R.m(f"e4/{t}/all/acc", 100 * base["acceptance"])
        R.m(f"e4/{t}/all/mc", 100 * base["mc_exposure"])
        R.m(f"e4/{t}/all/pruned", 100 * base["pruned_frac"])
        R.m(f"e4/{t}/all/log10R", base["full_log10"] - base["space_log10"], "{:.2f}")
        R.m(f"e4/{t}/seeds", n_seeds, "{:d}")
        for p in POLICIES:
            x = res.get((t, "Semantic-ILP", f"-{p}"))
            if not x:
                continue
            sd = sorted(set(x["seeds"]) & set(base["seeds"]))
            pr = paired([x["seeds"][s]["acceptance"] for s in sd], [base["seeds"][s]["acceptance"] for s in sd])
            prm = paired([x["seeds"][s]["mc_exposure"] for s in sd], [base["seeds"][s]["mc_exposure"] for s in sd])
            prc = paired([x["seeds"][s]["carbon_per_acc_g"] for s in sd],
                         [base["seeds"][s]["carbon_per_acc_g"] for s in sd])
            d_acc, d_en, d_co, d_lat, d_mc, d_sp = cells(x, base)
            rank[p].append(d_acc)
            for k, v, f in (("dacc", d_acc, "{:+.1f}"), ("den", d_en, "{:+.1f}"), ("dco", d_co, "{:+.1f}"),
                            ("dlat", d_lat, "{:+.1f}"), ("dmc", d_mc, "{:+.1f}"), ("dspace", d_sp, "{:+.2f}"),
                            ("p", pr["p_wilcoxon"], "{:.3f}"), ("pmc", prm["p_wilcoxon"], "{:.3f}"),
                            ("pco", prc["p_wilcoxon"], "{:.3f}")):
                R.m(f"e4/{t}/{p}/{k}", v, f)
            for k, v in (("dacc", d_acc), ("den", d_en), ("dco", d_co), ("dlat", d_lat), ("dmc", d_mc)):
                R.m(f"e4/{t}/{p}/{k}/abs", abs(v))
            a, b, c = sh(t, p)
            lines.append(f"\\quad without {p} & {a} & {b} & {c} & {d_acc:+.1f}{pstars(pr['p_wilcoxon'])} & "
                         f"{d_en:+.1f} & {d_co:+.1f}{pstars(prc['p_wilcoxon'])} & {d_lat:+.1f} & "
                         f"{d_mc:+.1f}{pstars(prm['p_wilcoxon'])} & {d_sp:+.2f} \\\\")
        none = res.get((t, "Semantic-ILP", "none"))
        if none:
            sd = sorted(set(none["seeds"]) & set(base["seeds"]))
            pr = paired([none["seeds"][s]["acceptance"] for s in sd], [base["seeds"][s]["acceptance"] for s in sd])
            d_acc, d_en, d_co, d_lat, d_mc, d_sp = cells(none, base)
            for k, v, f in (("acc", 100 * none["acceptance"], "{:.1f}"), ("dacc", d_acc, "{:+.1f}"),
                            ("den", d_en, "{:+.1f}"), ("dco", d_co, "{:+.1f}"), ("dlat", d_lat, "{:+.1f}"),
                            ("dmc", d_mc, "{:+.1f}"), ("dspace", d_sp, "{:+.2f}")):
                R.m(f"e4/{t}/none/{k}", v, f)
            for k, v in (("dacc", d_acc), ("den", d_en), ("dco", d_co), ("dlat", d_lat), ("dmc", d_mc)):
                R.m(f"e4/{t}/none/{k}/abs", abs(v))
            a, b, c = sh(t, "any")
            lines.append(f"\\quad without any policy & {a} & -- & {c} & {d_acc:+.1f}{pstars(pr['p_wilcoxon'])} & "
                         f"{d_en:+.1f} & {d_co:+.1f} & {d_lat:+.1f} & {d_mc:+.1f} & {d_sp:+.2f} \\\\")
        for a_ in ("Semantic-ILP", "Greedy"):
            ag, sem = res.get((t, a_, "agnostic")), res.get((t, a_, "all"))
            if ag and sem:
                R.m(f"e4/{t}/{a_}/agnostic/acc", 100 * ag["acceptance"])
                R.m(f"e4/{t}/{a_}/semantic/acc", 100 * sem["acceptance"])
                R.m(f"e4/{t}/{a_}/agnostic/violations", 100 * ag["policy_rej"])
                R.m(f"e4/{t}/{a_}/agnostic/dacc", 100 * (ag["acceptance"] - sem["acceptance"]), "{:+.1f}")
                R.m(f"e4/{t}/{a_}/agnostic/dacc/abs", abs(100 * (ag["acceptance"] - sem["acceptance"])))
                R.m(f"e4/{t}/{a_}/agnostic/en", ag["energy_per_acc_wh"], "{:.2f}")
                R.m(f"e4/{t}/{a_}/semantic/en", sem["energy_per_acc_wh"], "{:.2f}")
    head = ("\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{Activity with all policies (E4b)} & "
            "\\multicolumn{6}{c}{Change when the policy type is removed (E4)} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-10}\n"
            "Policy type & Pairs (\\%) & Only this (\\%) & Decisive (\\%) & $\\Delta$Acc.\\ (pp) & "
            "$\\Delta$Energy (\\%) & $\\Delta$CO$_2$ (\\%) & $\\Delta$Latency (\\%) & $\\Delta$MC exp.\\ (pp) & "
            "$\\Delta\\log_{10}\\prod_i|\\Cset_i|$ \\\\\n")
    body = head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    (R.tex / "tables" / "tab_e4_policies.tex").write_text(body, encoding="utf-8")
    order = sorted(rank, key=lambda p: -np.mean(rank[p]))
    for k, p in enumerate(order):
        R.m(f"e4/rank/{k + 1}", p)
        R.m(f"e4/rankval/{k + 1}", float(np.mean(rank[p])), "{:+.1f}")


# ====================================================================== E5 scalability
def e5(R):
    rows = load_runs("e5_scalability")
    if not rows:
        return
    byN = collections.defaultdict(list)
    for r in rows:
        byN[r["N"]].append(r)
    Ns = sorted(byN)
    tab, xs_tri, ys_tri = [], [], []
    for N in Ns:
        rs = byN[N]
        req = [q for r in rs for q in r["requests"]]
        rdf = [r for r in rs if "rdf_abox" in r]
        for r in rdf:
            xs_tri.append(N)
            ys_tri.append(r["rdf_abox"])
        mean = lambda k: float(np.mean([r[k] for r in rdf])) if rdf else np.nan

        def gap(key):
            v = [(q[key] - q["ilp_J"]) / abs(q["ilp_J"]) for q in req if q.get("ilp_ok") and q.get(key) is not None]
            return 100 * float(np.mean(v)) if v else np.nan

        def solved(key):
            v = [q.get(key) is not None for q in req if q.get("ilp_ok")]
            return 100 * float(np.mean(v)) if v else np.nan
        ilp_t = [q["ilp_t"] for q in req if q.get("ilp_status") not in ("infeasible_domain",)]
        st = dict(abox=mean("rdf_abox"), closure=mean("rdf_closure"), t_rdf=mean("rdf_t_total"),
                  t_closure=mean("rdf_t_closure"), t_rules_rdf=mean("rdf_t_rules"), t_val=mean("rdf_t_validate"),
                  t_abox=mean("rdf_t_abox"), t_extract=mean("rdf_t_extract"),
                  rules_ms=float(np.mean([q["t_rules_ms"] for q in req])),
                  ilp_mean=float(np.mean(ilp_t)), ilp_med=float(np.median(ilp_t)),
                  ilp_p95=float(np.percentile(ilp_t, 95)), ilp_max=float(np.max(ilp_t)),
                  sga_mean=float(np.mean([q["Semantic-GA_t"] for q in req])),
                  sga_max=float(np.max([q["Semantic-GA_t"] for q in req])),
                  ga_mean=float(np.mean([q["GA_t"] for q in req])),
                  greedy_ms=1e3 * float(np.mean([q["Greedy_t"] for q in req])),
                  gap_sga=gap("Semantic-GA_J"), gap_ga=gap("GA_J"), gap_greedy=gap("Greedy_J"),
                  solved_sga=solved("Semantic-GA_J"), solved_ga=solved("GA_J"), solved_greedy=solved("Greedy_J"),
                  ilp_tl=100 * float(np.mean([q.get("ilp_status") == "time_limit" for q in req])),
                  apsp=float(np.mean([r["t_apsp"] for r in rs])),
                  red=float(np.mean([q["phys_log10"] - q["domain_log10"] for q in req])),
                  pruned=100 * float(np.mean([q["pruned_frac"] for q in req])),
                  n_req=len(req), n_seeds=len(rs), equal=sum(r.get("rdf_equal", 0) for r in rdf),
                  checked=sum(len(r["requests"]) for r in rdf))
        ints = ("abox", "closure", "n_req", "n_seeds", "equal", "checked")
        for k, v in st.items():
            R.m(f"e5/{N}/{k}", v, "{:.0f}" if k in ints else ("{:.3f}" if k == "rules_ms" else "{:.2f}"))
        tab.append((N, st))
    # per-policy cost of the RDF rule evaluation and number of inferred pairs (reply to R1.16)
    from semvnf.rules import POLICIES
    for N in Ns:
        rdf = [r for r in byN[N] if "rdf_rule_time" in r]
        if not rdf:
            continue
        for p in POLICIES:
            R.m(f"e5/{N}/rule_t/{p}", float(np.mean([r["rdf_rule_time"].get(p, 0.0) for r in rdf])), "{:.2f}")
            R.m(f"e5/{N}/rule_pairs/{p}", float(np.mean([r["rdf_per_policy"].get(p, 0) for r in rdf])), "{:.0f}")
    sizes = [N for N in (50, 200, 1000) if any("rdf_rule_time" in r for r in byN.get(N, []))]
    if sizes:
        lines = []
        for p in POLICIES:
            cells = []
            for N in sizes:
                rdf = [r for r in byN[N] if "rdf_rule_time" in r]
                cells.append(f"{np.mean([r['rdf_per_policy'].get(p, 0) for r in rdf]):.0f}")
                cells.append(f"{np.mean([r['rdf_rule_time'].get(p, 0.0) for r in rdf]):.2f}")
            lines.append(f"{p} & " + " & ".join(cells) + " \\\\")
        head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$N={N}$}}" for N in sizes)
        sub = " & ".join("pairs & time (s)" for _ in sizes)
        body = ("\\begin{tabular}{@{}l" + "rr" * len(sizes) + "@{}}\n\\toprule\nPolicy & " + head + " \\\\\n & " + sub
                + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
        (R.tex / "tables" / "tab_policy_cost.tex").write_text(body, encoding="utf-8")
    coef = None
    if len(set(xs_tri)) > 1:
        A = np.vstack([np.array(xs_tri, float), np.ones(len(xs_tri))]).T
        coef, *_ = np.linalg.lstsq(A, np.array(ys_tri, float), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((np.array(ys_tri) - pred) ** 2))
        ss_tot = float(np.sum((np.array(ys_tri) - np.mean(ys_tri)) ** 2))
        R.m("e5/triples_per_node", coef[0], "{:.1f}")
        R.m("e5/triples_intercept", coef[1], "{:.0f}")
        R.m("e5/triples_r2", 1 - ss_res / ss_tot if ss_tot > 0 else 1.0, "{:.4f}")
    lines = [f"{N} & {st['abox']:.0f} & {st['t_rdf']:.1f} & {st['rules_ms']:.3f} & {st['ilp_med']:.2f} / {st['ilp_max']:.1f} & "
             f"{st['sga_mean']:.2f} & {st['ga_mean']:.2f} & {st['greedy_ms']:.2f} & {st['gap_sga']:.2f} & {st['gap_ga']:.2f} & "
             f"{st['gap_greedy']:.1f} \\\\" for N, st in tab]
    body = ("\\begin{tabular}{@{}rrrrrrrrrrr@{}}\n\\toprule\n$N$ & ABox triples & RDF pipeline (s) & Compiled rules (ms) & S-ILP median/max (s) & "
            "S-GA (s) & GA (s) & Greedy (ms) & Gap S-GA (\\%) & Gap GA (\\%) & Gap Greedy (\\%) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (R.tex / "tables" / "tab_e5_scalability.tex").write_text(body, encoding="utf-8")
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 2.9))
    ax = axes[0]
    ax.scatter(xs_tri, ys_tri, s=12, color="#0072B2")
    if coef is not None:
        xx = np.linspace(min(xs_tri), max(xs_tri), 10)
        ax.plot(xx, coef[0] * xx + coef[1], "k--", lw=1, label=f"{coef[0]:.1f} triples/node")
        ax.legend(frameon=False)
    ax.set_xlabel("Nodes $N$")
    ax.set_ylabel("ABox triples per epoch")
    ax.set_title("(a) Knowledge-graph size")
    ax = axes[1]
    tri = [st["abox"] for _, st in tab]
    for key, lab, mk in (("t_val", "SHACL validation", "o"), ("t_closure", "OWL 2 RL closure", "s"),
                         ("t_rules_rdf", "SPARQL rules", "^"), ("t_rdf", "full RDF pipeline", "D")):
        ax.plot(tri, [st[key] for _, st in tab], marker=mk, label=lab)
    ax.plot(tri, [st["rules_ms"] / 1e3 for _, st in tab], marker="x", color="black", label="compiled rules")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ABox triples")
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Reasoning time")
    ax.legend(frameon=False, fontsize=6.3)
    ax = axes[2]
    xsN = [N for N, _ in tab]
    ax.plot(xsN, [st["ilp_med"] for _, st in tab], marker="o", color=COL["Semantic-ILP"], label="S-ILP median")
    ax.plot(xsN, [st["ilp_max"] for _, st in tab], marker="o", ls=":", color=COL["Semantic-ILP"], label="S-ILP max")
    ax.plot(xsN, [st["sga_mean"] for _, st in tab], marker="s", color=COL["Semantic-GA"], label="S-GA")
    ax.plot(xsN, [st["ga_mean"] for _, st in tab], marker="^", color=COL["GA"], label="GA")
    ax.plot(xsN, [st["greedy_ms"] / 1e3 for _, st in tab], marker="D", color=COL["Greedy"], label="Greedy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes $N$")
    ax.set_ylabel("Decision time (s)")
    ax.set_title("(c) Solver time")
    ax.legend(frameon=False, fontsize=6.3)
    ax = axes[3]
    ax.plot(xsN, [st["gap_sga"] for _, st in tab], marker="s", color=COL["Semantic-GA"], label="S-GA")
    ax.plot(xsN, [st["gap_ga"] for _, st in tab], marker="^", color=COL["GA"], label="GA")
    ax.plot(xsN, [st["gap_greedy"] for _, st in tab], marker="D", color=COL["Greedy"], label="Greedy")
    ax.set_xscale("log")
    ax.set_xlabel("Nodes $N$")
    ax.set_ylabel("Gap to S-ILP optimum (%)")
    ax.set_title("(d) Solution quality")
    ax.legend(frameon=False, fontsize=6.3)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    R.save(fig, "fig_scalability")


# ====================================================================== E7 robustness
def _e7_stats(rs):
    fn = float(np.mean([r["fn_pairs"] for r in rs]))
    fp = float(np.mean([r["fp_pairs"] for r in rs]))
    viol = 100 * float(np.mean([r["reasons"].get("policy", 0) / max(r["arrivals"], 1) for r in rs]))
    acc = 100 * float(np.mean([r["acceptance"] for r in rs]))
    return fn, fp, viol, acc


def e7(R):
    rows = load_runs("e7_robustness")
    if not rows:
        return
    rows_c = load_runs("e7b_conservative")
    g = R.by(rows, "topo", "noise", "noise_kind")
    gc = R.by(rows_c, "topo", "noise", "noise_kind") if rows_c else {}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for t in TOPOS:
        rates = sorted({k[1] for k in g if k[0] == t})
        c = "#0072B2" if t == "euro28" else "#D55E00"
        for kind, ls in (("mixed", "-"), ("stale", "--"), ("wrong", ":")):
            pts = []
            for rt in rates:
                key = (t, rt, kind if rt > 0 else "mixed")
                if key not in g:
                    continue
                fn, fp, viol, acc = _e7_stats(g[key])
                pts.append((rt, fn, fp, viol, acc))
                for k, v, f in (("fn", fn, "{:.2f}"), ("fp", fp, "{:.2f}"), ("viol", viol, "{:.1f}"), ("acc", acc, "{:.1f}")):
                    R.m(f"e7/{t}/{kind}/{rt:g}/{k}", v, f)
                if rt > 0:
                    base = _e7_stats(g[(t, 0.0, "mixed")]) if (t, 0.0, "mixed") in g else None
                    if base:
                        R.m(f"e7/{t}/{kind}/{rt:g}/dacc", acc - base[3], "{:+.1f}")
                        R.m(f"e7/{t}/{kind}/{rt:g}/dacc/abs", abs(acc - base[3]))
            if not pts:
                continue
            P = np.array(pts)
            lab = f"{TOPO_LABEL[t]}, {kind}"
            axes[0].plot(100 * P[:, 0], P[:, 1], ls, color=c, marker="o", ms=3, label=lab)
            axes[0].plot(100 * P[:, 0], P[:, 2], ls, color=c, marker="x", ms=4)
            axes[1].plot(100 * P[:, 0], P[:, 3], ls, color=c, marker="o", ms=3, label=lab)
            axes[2].plot(100 * P[:, 0], P[:, 4], ls, color=c, marker="o", ms=3, label=lab)
        # conservative compilation (E7b): unknown attributes are denied
        for kind, mk in (("mixed", "s"), ("stale", "D")):
            pts = []
            for rt in sorted({k[1] for k in gc if k[0] == t and k[2] == kind}):
                fn, fp, viol, acc = _e7_stats(gc[(t, rt, kind)])
                pts.append((rt, fn, fp, viol, acc))
                for k, v, f in (("fn", fn, "{:.2f}"), ("fp", fp, "{:.2f}"), ("viol", viol, "{:.1f}"), ("acc", acc, "{:.1f}")):
                    R.m(f"e7b/{t}/{kind}/{rt:g}/{k}", v, f)
                if (t, rt, kind) in g:
                    pa = _e7_stats(g[(t, rt, kind)])
                    R.m(f"e7b/{t}/{kind}/{rt:g}/dacc_vs_permit", acc - pa[3], "{:+.1f}")
                    R.m(f"e7b/{t}/{kind}/{rt:g}/dacc_vs_permit/abs", abs(acc - pa[3]))
            if not pts:
                continue
            P = np.array(pts)
            kw = dict(color=c, marker=mk, ms=5, ls="none", mfc="none")
            axes[0].plot(100 * P[:, 0], P[:, 2], **kw)
            axes[1].plot(100 * P[:, 0], P[:, 3], label=f"{TOPO_LABEL[t]}, {kind}, deny", **kw)
            axes[2].plot(100 * P[:, 0], P[:, 4], **kw)
    axes[0].set_ylabel("Pairs per decision (o FN, x FP)")
    axes[1].set_ylabel("Proposals violating a policy (%)")
    axes[2].set_ylabel("Acceptance ratio (%)")
    for ax in axes:
        ax.set_xlabel("Knowledge error rate (%)")
        ax.grid(alpha=0.3)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    R.save(fig, "fig_e7_robustness")
    # compact table at 5 % and 20 %
    lines = []

    def cells(grp, key):
        if key not in grp:
            return ["--"] * 4
        fn, fp, viol, acc = _e7_stats(grp[key])
        return [f"{fn:.2f}", f"{fp:.2f}", f"{viol:.1f}", f"{acc:.1f}"]

    spec = [("none", "--", 0.0, "mixed", g)]
    for kind, label in (("stale", "stale"), ("wrong", "wrong"), ("mixed", "mixed")):
        for rt in (0.05, 0.2):
            spec.append((label, "permit", rt, kind, g))
            if kind != "wrong":
                spec.append((label, "deny", rt, kind, gc))
    for label, mode, rt, kind, grp in spec:
        row = [label, mode, f"{100 * rt:g}"]
        for t in TOPOS:
            row += cells(grp, (t, rt, kind))
        lines.append(" & ".join(row) + " \\\\")
    body = ("\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n & & & \\multicolumn{4}{c}{Euro28} & \\multicolumn{4}{c}{Germany50} \\\\\n"
            "\\cmidrule(lr){4-7}\\cmidrule(l){8-11}\n"
            "Errors & Unknown & $r$ (\\%) & FN & FP & Viol.\\ (\\%) & Acc.\\ (\\%) & FN & FP & Viol.\\ (\\%) & Acc.\\ (\\%) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (R.tex / "tables" / "tab_e7_robustness.tex").write_text(body, encoding="utf-8")


# ====================================================================== E8 carbon
def e8(R):
    rows = load_runs("e8_carbon")
    if not rows:
        return
    g = R.by(rows, "topo", "alg", "wC")

    def ci_of(rs, tpl):
        v = [r["per_template_ci"][tpl] for r in rs if tpl in r.get("per_template_ci", {})]
        return float(np.mean(v)) if v else float("nan")

    def lat_of(rs, tpl):
        v = [r["per_template_lat_sla"][tpl] for r in rs if tpl in r.get("per_template_lat_sla", {})]
        return float(np.mean(v)) if v else float("nan")

    lines = []
    for t in TOPOS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{TOPO_LABEL[t]}}}}} \\\\")
        for a in ("Semantic-ILP", "Semantic-GA"):
            base = g.get((t, a, 0.0))
            for wc in (0.0, 0.2, 0.5):
                rs = g.get((t, a, wc))
                if not rs:
                    continue
                acc = 100 * float(np.mean([r["acceptance"] for r in rs]))
                en = float(np.mean([r["energy_per_acc_wh"] for r in rs]))
                co = float(np.mean([r["carbon_per_acc_g"] for r in rs]))
                lat = float(np.mean([r["latency"] for r in rs]))
                ci_m, ci_u = ci_of(rs, "mIoT"), ci_of(rs, "URLLC")
                ci_all = float(np.mean([np.mean(list(r["per_template_ci"].values())) for r in rs
                                        if r.get("per_template_ci")])) if any(r.get("per_template_ci") for r in rs) else float("nan")
                dco = den = dlat = "--"
                if base and wc > 0:
                    bm = {r["job"]["seed"]: r for r in base}
                    cm = {r["job"]["seed"]: r for r in rs}
                    sd = sorted(set(bm) & set(cm))
                    pr = paired([cm[s]["carbon_per_acc_g"] for s in sd], [bm[s]["carbon_per_acc_g"] for s in sd])
                    pe = paired([cm[s]["energy_per_acc_wh"] for s in sd], [bm[s]["energy_per_acc_wh"] for s in sd])
                    pl = paired([cm[s]["latency"] for s in sd], [bm[s]["latency"] for s in sd])
                    pa = paired([cm[s]["acceptance"] for s in sd], [bm[s]["acceptance"] for s in sd])
                    # relative changes from the means rounded as displayed in the table (verifiable by the reader)
                    rd = lambda key, rows, nd, sc=1.0: round(sc * float(np.mean([rows[s][key] for s in sd])), nd)
                    b_co, c_co = rd("carbon_per_acc_g", bm, 2), rd("carbon_per_acc_g", cm, 2)
                    b_en, c_en = rd("energy_per_acc_wh", bm, 2), rd("energy_per_acc_wh", cm, 2)
                    b_lat, c_lat = rd("latency", bm, 1), rd("latency", cm, 1)
                    b_acc, c_acc = rd("acceptance", bm, 1, 100.0), rd("acceptance", cm, 1, 100.0)
                    v_co = 100 * (c_co - b_co) / b_co
                    v_en = 100 * (c_en - b_en) / b_en
                    v_lat = 100 * (c_lat - b_lat) / b_lat
                    v_acc = c_acc - b_acc
                    dco = f"{v_co:+.1f}{pstars(pr['p_wilcoxon'])}"
                    den = f"{v_en:+.1f}{pstars(pe['p_wilcoxon'])}"
                    dlat = f"{v_lat:+.1f}{pstars(pl['p_wilcoxon'])}"
                    for k, v, f in (("dco", v_co, "{:+.1f}"), ("den", v_en, "{:+.1f}"), ("dlat", v_lat, "{:+.1f}"),
                                    ("dacc", v_acc, "{:+.1f}"), ("pco", pr["p_wilcoxon"], "{:.3f}"),
                                    ("pen", pe["p_wilcoxon"], "{:.3f}"), ("plat", pl["p_wilcoxon"], "{:.3f}"),
                                    ("pacc", pa["p_wilcoxon"], "{:.3f}")):
                        R.m(f"e8/{t}/{a}/{wc:g}/{k}", v, f)
                    for k, v in (("dco", v_co), ("den", v_en), ("dlat", v_lat), ("dacc", v_acc)):
                        R.m(f"e8/{t}/{a}/{wc:g}/{k}/abs", abs(v))
                for k, v, f in (("acc", acc, "{:.1f}"), ("co", co, "{:.2f}"), ("en", en, "{:.2f}"), ("lat", lat, "{:.1f}"),
                                ("ci_miot", ci_m, "{:.0f}"), ("ci_urllc", ci_u, "{:.0f}"), ("ci_all", ci_all, "{:.0f}"),
                                ("latsla_miot", 100 * lat_of(rs, "mIoT"), "{:.0f}"),
                                ("latsla_urllc", 100 * lat_of(rs, "URLLC"), "{:.0f}")):
                    R.m(f"e8/{t}/{a}/{wc:g}/{k}", v, f)
                ci_m_s = "--" if not np.isfinite(ci_m) else f"{ci_m:.0f}"
                ci_u_s = "--" if not np.isfinite(ci_u) else f"{ci_u:.0f}"
                lines.append(f"{a} & {wc:g} & {acc:.1f} & {en:.2f} & {den} & {co:.2f} & {dco} & {lat:.1f} & {dlat} & "
                             f"{ci_m_s} / {ci_u_s} \\\\")
        # relative CI change of the delay-tolerant slice versus the latency-critical one (S-ILP, 0 -> 0.5)
        for a in ("Semantic-ILP", "Semantic-GA"):
            b, c = g.get((t, a, 0.0)), g.get((t, a, 0.5))
            if b and c:
                for tpl, key in (("mIoT", "miot"), ("URLLC", "urllc")):
                    x0, x1 = round(ci_of(b, tpl)), round(ci_of(c, tpl))      # as displayed (g/kWh, integers)
                    if np.isfinite(x0) and x0 > 0:
                        R.m(f"e8/{t}/{a}/dci_{key}", 100 * (x1 - x0) / x0, "{:+.1f}")
                        R.m(f"e8/{t}/{a}/dci_{key}/abs", abs(100 * (x1 - x0) / x0))
    body = ("\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\nMethod & $w_C$ & Acc.\\ (\\%) & Energy/acc.\\ (Wh) & $\\Delta$E (\\%) & "
            "CO$_2$/acc.\\ (g) & $\\Delta$CO$_2$ (\\%) & Latency (ms) & $\\Delta$Lat.\\ (\\%) & Host CI mIoT / URLLC (g/kWh) \\\\\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (R.tex / "tables" / "tab_e8_carbon.tex").write_text(body, encoding="utf-8")


# ====================================================================== E9 load sweep
def e9(R):
    rows = load_runs("e9_load")
    if not rows:
        return
    g = R.by(rows, "topo", "alg", "rho_n")
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2))
    for row, t in enumerate(TOPOS):
        for a in ALGS:
            rhos = sorted({k[2] for k in g if k[0] == t and k[1] == a})
            if not rhos:
                continue
            m_acc = [mean_ci([100 * r["acceptance"] for r in g[(t, a, rho)]]) for rho in rhos]
            m_en = [mean_ci([r["energy_per_acc_wh"] for r in g[(t, a, rho)]]) for rho in rhos]
            for rho, (mu, h) in zip(rhos, m_acc):
                R.m(f"e9/{t}/{a}/{rho:g}/acc", mu)
            for rho, (mu, h) in zip(rhos, m_en):
                R.m(f"e9/{t}/{a}/{rho:g}/en", mu, "{:.2f}")
            kw = dict(color=COL[a], marker=MRK[a], ms=4, lw=1.2, label=a, capsize=2)
            axes[row, 0].errorbar(rhos, [x[0] for x in m_acc], yerr=[x[1] for x in m_acc], **kw)
            axes[row, 1].errorbar(rhos, [x[0] for x in m_en], yerr=[x[1] for x in m_en], **kw)
        axes[row, 0].set_ylabel(f"{TOPO_LABEL[t]}\nAcceptance ratio (%)")
        axes[row, 1].set_ylabel("Energy per accepted SFC (Wh)")
    for ax in axes.ravel():
        ax.set_xlabel(r"Offered load of the normal state $\rho_N$")
        ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R.save(fig, "fig_e9_load")
    # paired differences Semantic-ILP vs Greedy per load level (acceptance in pp, energy per accepted SFC in %)
    for t in TOPOS:
        for rho in sorted({k[2] for k in g if k[0] == t}):
            a = {r["job"]["seed"]: r for r in g.get((t, "Semantic-ILP", rho), [])}
            b = {r["job"]["seed"]: r for r in g.get((t, "Greedy", rho), [])}
            sd = sorted(set(a) & set(b))
            if len(sd) < 3:
                continue
            pa = paired([a[s]["acceptance"] for s in sd], [b[s]["acceptance"] for s in sd])
            pe = paired([a[s]["energy_per_acc_wh"] for s in sd], [b[s]["energy_per_acc_wh"] for s in sd])
            dacc = 100 * float(np.mean([a[s]["acceptance"] - b[s]["acceptance"] for s in sd]))
            den = 100 * float(np.mean([(a[s]["energy_per_acc_wh"] - b[s]["energy_per_acc_wh"]) / b[s]["energy_per_acc_wh"]
                                       for s in sd]))
            for k, v, f in (("dacc", dacc, "{:+.1f}"), ("den", den, "{:+.1f}"), ("pacc", pa["p_wilcoxon"], "{:.3f}"),
                            ("pen", pe["p_wilcoxon"], "{:.3f}")):
                R.m(f"e9/{t}/ilp-greedy/{rho:g}/{k}", v, f)
            R.m(f"e9/{t}/ilp-greedy/{rho:g}/dacc/abs", abs(dacc))
            R.m(f"e9/{t}/ilp-greedy/{rho:g}/den/abs", abs(den))


# ====================================================================== E1-F failure scenario
def e1f(R):
    rows = load_runs("e1_fail")
    if not rows:
        return
    g = R.by(rows, "topo", "alg")
    lines = []
    for t in TOPOS:
        for a in ALGS:
            rs = g.get((t, a))
            if not rs:
                continue
            inter = sum(r["interrupted"] for r in rs)
            rec = sum(r["recovered"] for r in rs)
            acc, h = mean_ci([100 * r["acceptance"] for r in rs])
            en = float(np.mean([r["energy_per_acc_wh"] for r in rs]))
            R.m(f"e1f/{t}/{a}/recovered", 100 * rec / inter if inter else None)
            R.m(f"e1f/{t}/{a}/interrupted", inter / len(rs), "{:.1f}")
            R.m(f"e1f/{t}/{a}/acc", acc)
            R.m(f"e1f/{t}/{a}/acc/ci", h)
            lines.append(f"{TOPO_LABEL[t]} & {a} & {acc:.1f} $\\pm$ {h:.1f} & {inter / len(rs):.1f} & "
                         f"{(100 * rec / inter) if inter else float('nan'):.1f} & {en:.2f} \\\\")
    body = ("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nTopology & Method & Acceptance (\\%) & Interrupted SFCs/run & Recovered (\\%) & Energy/acc.\\ (Wh) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (R.tex / "tables" / "tab_e1fail.tex").write_text(body, encoding="utf-8")


EXTRA = [facts, drl_timing, fig_tradeoff, e2, e4, e5, e7, e8, e9, e1f]
