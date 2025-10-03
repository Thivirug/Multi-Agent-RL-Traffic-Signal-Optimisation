# Configurations for multi-agent environment (PettingZoo) and model training hyperparams
import os

# ! Training config vars
ARG_DICT = {
    "ppo" : {
        "n_iterations": 500,   
        "chkpoint_eval_freq": 10 # checkpointing and training eval freq
    },

    "dqn" : { 
        "n_iterations": 500,   
        "chkpoint_eval_freq": 10 # checkpointing and training eval freq
    },

    "sac": {
        "n_iterations": 400,   
        "chkpoint_eval_freq": 10 # checkpointing and training eval freq
    }
}

# ! ------- ENV CONFIG -------
# Parameters for sumo_rl.parallel_env
ENV_CONFIG = {
    'net_file': os.path.abspath('src/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml'),
    'route_file': os.path.abspath('src/sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml'),
    'out_csv_name': os.path.abspath("Code/outputs/logs/logs"),
    'use_gui': False,
    'num_seconds': 12000,      # Total simulation time
    'delta_time': 5,           # Length of a simulation step (seconds)
    'yellow_time': 2,          # Length of yellow phase
    'min_green': 6,            # Minimum green time
    'single_agent': False,     # Multi-agent setup
    'reward_fn': 'diff-waiting-time',  # computed based on changes during each delta_time window
    'add_per_agent_info': True,
    'add_system_info': True,   # Add system-wide information for centralized training
    'sumo_seed': 'random',     # Randomize traffic patterns # ! makes the algo more robust
}

# ! ------- Algorithm hyperparams -------

# PPO Hyperparameters 
PPO_hparams = { 
    'lr': [
        [0, 5e-4],         # <- initial learning rate at timestep 0
        [100000, 1e-4],    # <- learning rate at 100k timesteps
        [300000, 5e-5],    # <- learning rate at 300k timesteps
    ],            
    'train_batch_size_per_learner': 256,  # Batch size per learner
    'entropy_coeff' : [              # Entropy coefficient for exploration
        [0, 0.1],          # <- initial value at timestep 0
        [100000, 0.06],     # <- value at 100k timesteps
        [250000, 0.02],     # <- value at 250k timesteps
        [400000, 0.01],     # <- value at 400k timesteps
    ],       
    'kl_coeff': 0.2,               # KL divergence coefficient
    'clip_param': 0.2,             # PPO clipping parameter
    'vf_clip_param': 10.0,         # Value function clipping parameter
    'gamma': 0.99,                 # Discount factor
    'lambda_': 0.95,               # GAE lambda parameter
    'use_gae': True,               # Use Generalized Advantage Estimation
    'vf_loss_coeff': 0.5,          # Value function loss coefficient
    'grad_clip': 0.5,              # Gradient clipping
}

# DQN Hyperparameters  
DQN_hparams = {
    # Training configuration
    "train_batch_size_per_learner": 256,  # Batch size per learner (New API)
    'lr': [
        [0, 5e-4],         # <- initial learning rate at timestep 0
        [100000, 1e-4],    # <- learning rate at 100k timesteps
        [300000, 5e-5],    # <- learning rate at 300k timesteps
    ], 
    
    # Core DQN parameters
    "gamma": 0.99,                       # Discount factor
    "target_network_update_freq": 1000,  # Target network update frequency (in env steps)
    "n_step": 1,                         # Use 1-step returns (fixes scipy buffer issue)
    
    # Network architecture
    "double_q": True,                    # Use double Q-learning
    "dueling": True,                     # Use dueling network architecture
    
    # Replay buffer settings (for new API stack)
    # Using non-prioritized buffer for stability in long training runs
    "replay_buffer_config": {
        "type": "MultiAgentEpisodeReplayBuffer",  # Non-prioritized - more stable
        "capacity": 500000,              # Large enough for ~50 episodes (4 agents × 2400 steps each)
    },
    
    "num_steps_sampled_before_learning_starts": 5000,  # ~2 episodes before training starts
 
}

# SAC Hyperparameters
SAC_hparams = {
    "twin_q": True,                # Use two Q-networks for the critics, to mitigate overestimation bias
    "gamma": 0.99,                 # discount factor
    "tau" : 0.005,                 # Soft update coefficeint
    "train_batch_size_per_learner" : 256, 
    "num_steps_sampled_before_learning_starts": 5000, # number of steps to collect before training starts
    "target_entropy": "auto",      # automatically tune the entropy coefficeint (alpha)
    "target_network_update_freq": 1, # update target network every 'train_batch_size' steps. 1 is standards for soft updates
    "optimization_config": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    }
}
