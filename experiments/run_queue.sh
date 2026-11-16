#!/usr/bin/env bash
# Sequential execution of all experiments of the revised paper (resumable).
# Usage: bash run_queue.sh [workers]
set -u
W=${1:-5}
cd "$(dirname "$0")"
R=../results
wait_for () {  # wait until file $1 contains string $2
  until grep -q "$2" "$1" 2>/dev/null; do sleep 30; done
}
# DRL agents must be trained first (train_drl.py writes <topo>_<alg>.json at the end)
for t in euro28 germany50; do
  for a in DDPG GCN-PPO; do
    until python -c "import json,sys; d=json.load(open('$R/drl/${t}_${a}.json')); sys.exit(0 if d['n_traces']>=30 else 1)" 2>/dev/null; do sleep 30; done
  done
done
echo "[queue] DRL agents ready $(date)"
wait_for $R/e1_online.log "300/300"      # the first (non-DRL) part of E1 may still be running
echo "[queue] first part of e1 finished $(date)"
python e1_online.py --workers $W            >> $R/e1_online.log 2>&1
echo "[queue] e1 done $(date)"
python e1_online.py --workers $W --scenario fail --algs Semantic-ILP,Semantic-GA,GA,Greedy,DDPG,GCN-PPO > $R/e1_fail.log 2>&1
echo "[queue] e1_fail done $(date)"
python e2_shadow.py --workers $W            > $R/e2_shadow.log 2>&1
echo "[queue] e2 done $(date)"
python e4_policies.py --workers $W          > $R/e4_policies.log 2>&1
echo "[queue] e4 done $(date)"
python e4b_policy_shares.py --workers $W    > $R/e4b_policy_shares.log 2>&1
echo "[queue] e4b done $(date)"
python e7_robustness.py --workers $W        > $R/e7_robustness.log 2>&1
echo "[queue] e7 done $(date)"
python e7b_conservative.py --workers $W     > $R/e7b_conservative.log 2>&1
echo "[queue] e7b done $(date)"
python e8_carbon.py --workers $W            > $R/e8_carbon.log 2>&1
echo "[queue] e8 done $(date)"
python e9_load.py --workers $W              > $R/e9_load.log 2>&1
echo "[queue] e9 done $(date)"
python e5_scalability.py --workers 3        > $R/e5_scalability.log 2>&1
echo "[queue] e5 done $(date)"
# decision times of all methods under identical, low-load conditions (nothing else may run)
python e1_timing.py --workers 3             > $R/e1_timing.log 2>&1
echo "[queue] e1_timing done $(date)"
echo "[queue] ALL DONE $(date)"
