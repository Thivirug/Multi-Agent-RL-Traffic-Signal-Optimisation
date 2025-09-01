# Configurations for multi-agent environment (Pettingzoo) and model training hyperparams
import os

# ! Other config vars needed
n_iterations = 4 # num of training iterations
checkpoint_freq = 2 # checkpointing freq

# ! ------- ENV CONFIG -------
# Parameters in sumo_rl.parallel_env 
# TODO : Get params starting from Line 93 in sumo-rl/sumo_rl/environment/env.py

ENV_CONFIG = {
    'net_file': "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml",
    'route_file': "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml",
    'out_csv_name': os.path.abspath("Code/outputs/logs/logs"),
    'use_gui': False,
    'num_seconds': 36000,
    # add more .. if needed
}

# ! ------- Algorithm hyperparams overrides -------

# PPO
PPO_hparams = {
    'lr': 0.0001,
    'train_batch_size': 4000,
    'entropy_coeff': 0.01,
    'kl_coeff':0.2,
    'clip_param':0.2,
    'vf_clip_param':10.0
}

# DQN 
DQN_hparams = {

}