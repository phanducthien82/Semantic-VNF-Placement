"""Remove the runs of given algorithms from an experiment (keeps a backup), so that they can be re-run.

Used after the DRL reward was corrected (headroom term added to the step reward, Section 4.1),
which required re-training DDPG/GCN-PPO and re-running their evaluation runs.

    python drop_runs.py e1_online DDPG,GCN-PPO
"""
import json
import shutil
import sys
import time

from common import RESULTS

if __name__ == "__main__":
    exp, algs = sys.argv[1], set(sys.argv[2].split(","))
    p = RESULTS / exp / "runs.jsonl"
    bak = p.with_suffix(f".jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy(p, bak)
    keep, drop = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("job", {}).get("alg") in algs:
            drop += 1
            continue
        keep.append(line)
    p.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    print(f"{exp}: dropped {drop} runs of {sorted(algs)}, kept {len(keep)}; backup {bak.name}")
