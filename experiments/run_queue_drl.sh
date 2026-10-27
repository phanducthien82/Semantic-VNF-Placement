#!/usr/bin/env bash
# Re-evaluation of the DRL baselines after the reward correction (headroom term added to the step
# reward, Section 4.1): the old DDPG/GCN-PPO runs were removed with drop_runs.py, the agents were
# re-trained with train_drl.py, and the E1/E1-F runs of DDPG and GCN-PPO are repeated here.
# Afterwards the policy-activity study E4b is run.
set -u
W=${1:-2}
cd "$(dirname "$0")"
R=../results
newer () {  # true if $1 was modified after the reference time stamp file $2
  [ "$1" -nt "$2" ]
}
python e1_online.py --workers $W --algs DDPG,GCN-PPO --topos euro28               >> $R/e1_drl_rerun.log 2>&1
python e1_online.py --workers $W --algs GCN-PPO --topos germany50                 >> $R/e1_drl_rerun.log 2>&1
python e1_online.py --workers $W --scenario fail --algs DDPG,GCN-PPO --topos euro28 >> $R/e1_drl_rerun.log 2>&1
python e1_online.py --workers $W --scenario fail --algs GCN-PPO --topos germany50   >> $R/e1_drl_rerun.log 2>&1
# the Germany50 DDPG agent is the last one to finish re-training
until newer $R/drl/germany50_DDPG.json $R/drl/germany50_GCN-PPO.json; do sleep 20; done
echo "[drl] germany50 DDPG ready $(date)" >> $R/e1_drl_rerun.log
python e1_online.py --workers $W --algs DDPG --topos germany50                    >> $R/e1_drl_rerun.log 2>&1
python e1_online.py --workers $W --scenario fail --algs DDPG --topos germany50      >> $R/e1_drl_rerun.log 2>&1
echo "[drl] E1/E1-F re-runs done $(date)" >> $R/e1_drl_rerun.log
python e4b_policy_shares.py --workers $W                                            > $R/e4b_policy_shares.log 2>&1
echo "[drl] E4b done $(date)" >> $R/e1_drl_rerun.log
