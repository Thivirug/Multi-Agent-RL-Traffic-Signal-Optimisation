
#### Tentative Project Directory Structure

```
project_root/
├── code/                  # Custom codebase 
│   ├── experiments/       # Scripts for training/evaluation (mirrors sumo-rl's experiments/)
│   │   ├── algorithms/    # RL-algorithms used to train agents (DQN, PPO, etc)
│   │   ├── train_multiagent.py  # Main training script 
│   │   └── evaluate.py    # Script for testing trained models
│   ├── nets/              # Custom SUMO networks/routes (extend sumo-rl's nets/ if needed)
│   │   └── my_network/    # Example custom network dir
│   │       ├── my_net.net.xml  # Generated via NETEDIT or netconvert
│   │       └── my_routes.rou.xml  # Vehicle flows 
│   ├── notebooks/         # jupyter notebooks for exploration and visualization
│   ├── utils/             # Optional: Custom utilities (e.g., reward/observation functions)
│   │   └── custom_rewards.py  # If overriding defaults
│   ├── outputs/           # Logs, checkpoints, metrics (mirrors sumo-rl's outputs/)
│   │   ├── checkpoints/   # Model saves
│   │   └── logs/          # CSV metrics from sumo-rl env
│   ├── config.py          # Shared hyperparameters/configs (e.g., env paths, RL settings)
│   ├── initial_testing/   # Preliminary testing scripts (GPU, GUI, Env props, etc) 
├── sumo-rl/               # Cloned repo (install as package; don't add code here)
├── sumoenv/               # Virtual env (activate for all work)
├── .gitignore
├── mac_requirements.txt   # Platform-specific reqs
├── README.md              # Update with project notes
└── requirements.txt       # Core dependencies (e.g., ray[rllib], gymnasium)
```

My Setup differnces
- since using windows OS I am unable to run cuda version that supports uvlib in pytorch this disallows me to use:
    .env_runners(num_env_runners=1, num_gpus_per_env_runner=1)
    .learners(num_learners=2, num_cpus_per_learner=5)
so cant train in parellel. However i instead just use:
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .env_runners(num_env_runners=1, num_gpus_per_env_runner=1)
    .learners(num_learners=1, num_cpus_per_learner=4, num_gpus_per_learner=1)
one single thread learner.