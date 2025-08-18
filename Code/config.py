# Configurations for multi-agent environment (Pettingzoo) and model training hyperparams

# ! Other config vars needed
n_iterations = 1000 # num of training iterations
checkpoint_freq = 10 # checkpointing freq

# ! ------- ENV CONFIG -------
# Parameters in sumo_rl.parallel_env 
# TODO : Get params starting from Line 93 in sumo-rl/sumo_rl/environment/env.py

ENV_CONFIG = {
    'net_file': "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml",
    'route_file': "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml",
    'out_csv_name': "outputs/logs",
    'use_gui': False,
    'num_seconds': 36000,
    # add more .. if needed
}

# ! ------- Algorithm hyperparams overrides -------

# PPO
PPO_hparams = {

}

# DQN 
DQN_hparams = {

}