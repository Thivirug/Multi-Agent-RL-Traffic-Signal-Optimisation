# # Configurations for multi-agent environment (Pettingzoo) and model training hyperparams
# import os

# # ! Other config vars needed
# n_iterations = 4 # num of training iterations
# checkpoint_freq = 2 # checkpointing freq

# # ! ------- ENV CONFIG -------
# # Parameters in sumo_rl.parallel_env 
# # TODO : Get params starting from Line 93 in sumo-rl/sumo_rl/environment/env.py

# ENV_CONFIG = {
#     'net_file': "src/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml",
#     'route_file': "src/sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml",
#     'out_csv_name': os.path.abspath("Code/outputs/logs/logs"),
#     'use_gui': False,
#     'num_seconds': 36000,
#     # add more .. if needed
# }

# # ! ------- Algorithm hyperparams overrides -------

# # PPO
# PPO_hparams = {
#     'lr': [
#         [0, 1e-5],  # <- initial value at timestep 0
#         [100000, 1e-2],  # <- final value at .. timesteps
#     ],
#     'train_batch_size_per_learner': 512,
#     'entropy_coeff': 0.01,
#     'kl_coeff':0.2,
#     'clip_param':0.2,
#     'vf_clip_param':10.0,
#     'gamma': 0.9
# }

# # DQN 
# DQN_hparams = {

# }

# # Configurations for multi-agent environment (Pettingzoo) and model training hyperparams
# import os

# # ! Training config vars
# n_iterations = 100 # Increased for more meaningful training
# checkpoint_freq = 10 # Checkpointing frequency

# # ! ------- ENV CONFIG -------
# # Parameters for sumo_rl.parallel_env
# # NOTE: Make sure the paths to net_file and route_file are correct for your project structure.
# # This config assumes a 'nets' folder in your project root.
# ENV_CONFIG = {
#     'net_file': 'src/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml',
#     'route_file': 'src/sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml',
#     'out_csv_name': os.path.abspath("outputs/logs/logs"),
#     'use_gui': False,
#     'num_seconds': 80000,      # Total simulation time
#     'delta_time': 5,           # Length of a simulation step (seconds)
#     'yellow_time': 2,          # Length of yellow phase
#     'min_green': 5,            # Minimum green time
#     'max_green': 60,           # Maximum green time
#     'single_agent': False,
#     'reward_fn': 'diff-waiting-time', # A common reward function for traffic optimization
#     'add_per_agent_info': True,
# }

# # ! ------- Algorithm hyperparams -------

# # PPO Hyperparameters for CTDE
# PPO_hparams = {
#     # 'lr': [
#     #     [0, 1e-5],  # <- initial value at timestep 0
#     #     [100000, 1e-2],  # <- final value at .. timesteps
#     # ],
#     'lr': 1e-4,
#     'train_batch_size_per_learner': 512,
#     'entropy_coeff': 0.01,
#     'kl_coeff':0.2,
#     'clip_param':0.2,
#     'vf_clip_param':10.0,
#     'gamma': 0.9
# }

# # DQN Hyperparameters (for Independent DQN with shared policy)
# DQN_hparams = {
#     "learning_starts": 1000,
#     "train_batch_size": 32,
#     "lr": 1e-4,
#     "gamma": 0.99,
#     "replay_buffer_config": {
#         "type": "MultiAgentReplayBuffer",
#         "capacity": 50000,
#     },
#     "exploration_config": {
#         "type": "EpsilonGreedy",
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.02,
#         "epsilon_timesteps": 100000,
#     },
# }

# Configurations for multi-agent environment (PettingZoo) and model training hyperparams
import os

# ! Training config vars
n_iterations = 200  # Increased for more meaningful training
checkpoint_freq = 10  # Checkpointing and eval frequency in training

# ! ------- ENV CONFIG -------
# Parameters for sumo_rl.parallel_env
ENV_CONFIG = {
    'net_file': os.path.abspath('src/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml'),
    'route_file': os.path.abspath('src/sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml'),
    'out_csv_name': os.path.abspath("Code/outputs/logs/logs"),
    'use_gui': False,
    'num_seconds': 3600,      # Total simulation time
    'delta_time': 5,           # Length of a simulation step (seconds)
    'yellow_time': 3,          # Length of yellow phase
    'min_green': 8,            # Minimum green time
    'max_green': 60,           # Maximum green time
    'single_agent': False,     # Multi-agent setup
    'reward_fn': 'diff-waiting-time',  # computed based on changes during each delta_time window
    'add_per_agent_info': True,
    'add_system_info': True,   # Add system-wide information for centralized training
    'sumo_seed': 'random',     # Randomize traffic patterns # ! makes the algo more robust
}

# ! ------- Algorithm hyperparams -------

# PPO Hyperparameters for CTDE
PPO_hparams = {
    'lr':2e-5,                    # Learning rate
    'train_batch_size_per_learner': 256,  # Batch size per learner
    'entropy_coeff': 0.1,         # Entropy coefficient for exploration
    'kl_coeff': 0.2,               # KL divergence coefficient
    'clip_param': 0.2,             # PPO clipping parameter
    'vf_clip_param': 10.0,         # Value function clipping parameter
    'gamma': 0.99,                 # Discount factor
    'lambda_': 0.95,               # GAE lambda parameter
    'use_gae': True,               # Use Generalized Advantage Estimation
    'vf_loss_coeff': 0.5,          # Value function loss coefficient
    'grad_clip': 0.5,              # Gradient clipping
}

# DQN Hyperparameters for Independent Learning
DQN_hparams = {
    "learning_starts": 5000,       # Start training after this many steps
    "train_batch_size": 32,        # Batch size for training
    "lr": 1e-4,                    # Learning rate
    "gamma": 0.99,                 # Discount factor
    "target_network_update_freq": 1000,  # Target network update frequency
    "replay_buffer_config": {
        "type": "MultiAgentReplayBuffer",
        "capacity": 100000,        # Replay buffer size
    },
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,    # Initial exploration rate
        "final_epsilon": 0.02,     # Final exploration rate
        "epsilon_timesteps": 200000,  # Steps to decay epsilon
    },
    "double_q": True,              # Use double Q-learning
    "dueling": True,               # Use dueling network architecture
    "n_step": 1,                   # N-step returns
}

# Algorithm selection mapping
ALGO_HPARAMS = {
    'ppo': PPO_hparams,
    'dqn': DQN_hparams,
}