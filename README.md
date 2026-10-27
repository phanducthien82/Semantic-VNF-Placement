# Semantic-aware VNF placement: knowledge graph + hybrid optimisation

Reference implementation and complete experimental artefact of the revised manuscript

> D.-T. Phan, K. Li, N. D. To, *A Cognitive Orchestration Agent for VNF Placement: Bridging Semantic
> Graphs and Hybrid Optimization in Sustainable 5G/6G Networks*, Computer Networks (COMNET-D-26-06541,
> revision 1).

Every number, table and figure of the revised paper is produced by the scripts of this repository
from the raw per-run results stored in `results/` (see *Reproducing the paper*).

## Contents

| Path | Content |
|---|---|
| `ontology/nfv-infra.ttl` | TBox v2.0 (OWL 2 RL constructs only) |
| `ontology/nfv-shapes.ttl` | SHACL shapes of the validation stage |
| `ontology/policies.swrl` | the seven policy types P1-P7 as DL-safe SWRL rules (documentation form) |
| `semvnf/kb.py` | reference RDF pipeline: ABox generation, SHACL + freshness validation, OWL 2 RL closure (owlrl), policy rules as SPARQL CONSTRUCT, SPARQL extraction |
| `semvnf/rules.py` | compiled (vectorised) evaluation of the same rules, used per arrival; equivalence with `kb.py` is checked in the experiments |
| `semvnf/model.py` | single-request problem, objective coefficients, latency decomposition, admission validator |
| `semvnf/ilp.py` | Semantic-ILP: arc-flow MILP solved with HiGHS (or Gurobi) |
| `semvnf/ga.py` | Semantic-GA and its ablation variants (vectorised) |
| `semvnf/routing.py` | latency-weighted shortest paths, LFA and Yen candidate path sets |
| `semvnf/pricing.py` | policy-scarcity pricing |
| `semvnf/heuristics.py`, `semvnf/drl.py` | Greedy, Random, DDPG and GCN-PPO baselines |
| `semvnf/sim.py`, `semvnf/shadow.py` | online event-driven simulator; shadow evaluation on identical epochs |
| `semvnf/topology.py`, `substrate.py`, `catalog.py`, `workload.py`, `noise.py` | topologies, attributes, SFC templates, MMPP workload, imperfect-knowledge model |
| `data/sndlib/` | SNDlib `nobel-eu` (Euro28) and `germany50` network files |
| `experiments/` | one script per experiment, `make_report.py` (tables/figures/numbers), `run_queue.sh` |
| `results/` | raw per-run results (`*/runs.jsonl`), `meta.json` (versions), DRL checkpoints, summaries |
| `legacy/` | ontology files of the first submission (v1.0), kept for reference only |

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1 (HiGHS), NetworkX 3.6.1, rdflib 7.6.0, owlrl 7.6.2,
pySHACL 0.40.1, PyTorch 2.14.1 (CPU), Matplotlib 3.11.2. Gurobi is optional (cross-check only).

## Reproducing the paper

All scripts are resumable; each run appends one JSON line to `results/<experiment>/runs.jsonl`.

```bash
cd experiments
python train_drl.py --topo euro28 --alg DDPG --traces 30     # DRL training (seeds >= 1000)
python train_drl.py --topo euro28 --alg GCN-PPO --traces 60  # (same for --topo germany50)
python e1_online.py                                # E0 + E1 main comparison, seeds 0-29
python e1_online.py --scenario fail                # E1-F node failures
python e2_shadow.py                                # E2/E3/E6/E10 on identical decision epochs
python e4_policies.py                              # E4 policy ablation
python e4b_policy_shares.py                        # E4b activity of each policy type
python e5_scalability.py                           # E5 Barabasi-Albert graphs up to 1000 nodes
python e7_robustness.py                            # E7 imperfect knowledge (default mode: unknown = permit)
python e7b_conservative.py                         # E7b imperfect knowledge, conservative mode (unknown = deny)
python e8_carbon.py                                # E8 energy-only vs carbon-aware
python e9_load.py                                  # E9 offered-load sweep
python e1_timing.py                                # E1-T decision times of all methods on an otherwise idle machine
python make_report.py --tex <path-to-latex-folder> # tables, figures and results_macros.tex
# or E1-E9 in order (after the DRL training):
bash run_queue.sh
```

Seeds: evaluation 0-29 (E1) and 0-9 (other experiments; 0-4 for E4b and E1-T); pilot calibration 900-905;
DRL training 1000+.

History of the result files: the DRL step reward first lacked the headroom term of the objective; after the
correction the agents were re-trained and their E1/E1-F runs repeated (`drop_runs.py`, `run_queue_drl.sh`).
The superseded runs are kept locally as `runs.jsonl.bak-*` (not part of the release).

## Licence

MIT (see `LICENSE`).
