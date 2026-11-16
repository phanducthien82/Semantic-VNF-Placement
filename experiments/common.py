"""Shared experiment infrastructure: resumable parallel job execution and metadata."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def job_id(job: dict) -> str:
    return "|".join(f"{k}={job[k]}" for k in sorted(job) if not k.startswith("_"))


def metadata() -> dict:
    import networkx
    import numpy
    import scipy
    meta = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                networkx=networkx.__version__, platform=platform.platform(), processor=platform.processor(),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import highspy
        meta["highspy"] = getattr(highspy, "__version__", "installed")
    except Exception:
        pass
    for mod in ("rdflib", "owlrl", "pyshacl", "torch", "gurobipy"):
        try:
            m = __import__(mod)
            meta[mod] = getattr(m, "__version__", "installed")
        except Exception:
            pass
    try:
        import subprocess
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        meta["git_commit"] = None
    return meta


def _safe(fn, job):
    t0 = time.perf_counter()
    try:
        res = fn(job)
        res = dict(res)
        res.update(job={k: v for k, v in job.items() if not k.startswith("_")}, id=job_id(job),
                   wall_s=time.perf_counter() - t0, status="ok")
        return res
    except Exception as e:  # keep the other jobs running
        return dict(job={k: v for k, v in job.items() if not k.startswith("_")}, id=job_id(job), status="error",
                    error=repr(e), tb=traceback.format_exc(), wall_s=time.perf_counter() - t0)


def run_jobs(exp: str, jobs: list, fn, workers: int = 4, out_name: str = "runs.jsonl") -> Path:
    out_dir = RESULTS / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    meta = out_dir / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps(metadata(), indent=2))
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["id"])
            except Exception:
                pass
    todo = [j for j in jobs if job_id(j) not in done]
    print(f"[{exp}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run, workers={workers}", flush=True)
    if not todo:
        return out
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        if workers <= 1:
            for k, j in enumerate(todo):
                r = _safe(fn, j)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"[{exp}] {k + 1}/{len(todo)} {r['id']} {r['status']} {r['wall_s']:.1f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_safe, fn, j): j for j in todo}
                for k, f in enumerate(as_completed(futs)):
                    r = f.result()
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    el = time.time() - t0
                    print(f"[{exp}] {k + 1}/{len(todo)} {r['id']} {r['status']} {r['wall_s']:.1f}s "
                          f"(elapsed {el / 60:.1f} min)", flush=True)
    return out


def load_runs(exp: str, out_name: str = "runs.jsonl") -> list:
    p = RESULTS / exp / out_name
    rows, seen = [], set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            # keep the first successful record of each job (a job may be recorded twice when two
            # runners of the same experiment overlap; both records come from the same seed)
            if r.get("status") == "ok" and r.get("id") not in seen:
                seen.add(r.get("id"))
                rows.append(r)
    return rows
