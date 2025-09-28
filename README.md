
# Multi-Agent RL Traffic Signal Optimisation

## Project Directory Structure

```
Multi-Agent-RL-Traffic-Signal-Optimisation/
├── 📁 Code/                              # Main application code
│   ├── 📄 config.py                      # Configuration settings
│   │
│   ├── 📁 experiments/                   # ML experiments and algorithms
│   │   ├── 📄 algorithms.py              # Algorithm implementations
│   │   ├── 📄 evaluate_multiagent.py     # Multi-agent evaluation scripts
│   │   ├── 📄 train_multiagent.py        # Multi-agent training scripts
│   │   └── 📄 todo.txt                   # Experiment tasks
│   │
│   ├── 📁 initial_testing/               # Environment and setup tests
│   │   ├── 📄 env_info.py                # Environment information
│   │   ├── 📄 test_gpu.py                # GPU testing (Linux)
│   │   ├── 📄 test_gpu_mac.py            # GPU testing (macOS)
│   │   ├── 📄 test_install.py            # Installation verification
│   │   └── 📄 test_mac.py                # macOS specific tests
│   │
│   ├── 📁 outputs/                       # Training outputs and results
│   │   ├── 📄 results_{algoname}.json    # {algoname} algorithm results
│   │   ├── 📄 todo.txt                   # Output tasks
│   │   │
│   │   ├── 📁 checkpoints/               # Model checkpoints
│   │   │   └── 📁 {algoname}/            # {algoname} model checkpoints
│   │   │       ├── 📁 10/ ... 📁 500/    # Checkpoint iterations
│   │   │
│   │   ├── 📁 logs/                      # Training logs
│   │   │   └── 📁 {algoname}/            # {algoname} training logs
│   │   │       └── 📄 logs_*.csv         # CSV log files
│   │   │
│   │   ├── 📁 plots/                     # Generated plots and visualizations
│   │   │   └── 📁 {algoname}/            # {algoname} algorithm plots
│   │   │       ├── 📁 rewards/           # Reward plots
│   │   │       └── 📁 waiting_time_multiEpisode/ # Multi-episode WT plots
│   │   │
│   │   └── 📁 recordings/                # Video recordings
│   │       └── 📁 {algoname}/            # {algoname} training recordings
│   │
│   └── 📁 utils/                         # Utility scripts
│       ├── 📄 record_eval.py             # Recording evaluation
│       ├── 📄 rewards_plot.py            # Reward plotting utilities
│       ├── 📄 WT_plot_multi_episode.py   # Multi-episode waiting time plots
│       ├── 📄 WT_plot_single_log.py      # Single log waiting time plots
│       └── 📄 todo.txt                   # Utility tasks
│
├── 📁 src/                               # Source code and dependencies
│   └── 📁 sumo-rl/                       # SUMO-RL environment library
│       ├── 📄 setup.py                   # Package setup
│       ├── 📄 README.md                  # Library documentation
│       ├── 📁 sumo_rl/                   # Core library code
│       ├── 📁 experiments/               # Library examples
│       ├── 📁 docs/                      # Documentation
│       └── 📁 tests/                     # Unit tests
│
├── 📁 sumo/                              # SUMO traffic simulator
├── 📁 sumo-rl/                           # SUMO-RL (alternative location)
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies (Linux)
├── 📄 mac_requirements.txt               # Python dependencies (macOS)
├── 📄 MARL_notes.md                      # Multi-Agent RL notes
├── 📄 notes.md                           # General project notes
├── 📄 ppo_working.md                     # PPO implementation notes
└── 📄 logs_desc.md                       # Log file descriptions
```

## Directory Descriptions

### 🔧 **Code/**
Main application directory containing all custom implementation code, experiments, and utilities.

### 🧪 **Code/experiments/**
Contains the core machine learning algorithms and training/evaluation scripts for multi-agent reinforcement learning.

### 🛠️ **Code/initial_testing/**
Testing scripts to verify environment setup, GPU functionality, and installation correctness across different platforms.

### 📊 **Code/outputs/**
All training outputs including model checkpoints, logs, visualizations, and recorded sessions organized by algorithm type.

### 🔨 **Code/utils/**
Utility scripts for data processing, visualization, and analysis of training results.

### 📦 **src/**
External dependencies and libraries, primarily the SUMO-RL environment for traffic simulation.

## Key Files

| File | Description |
|------|-------------|
| `config.py` | Central configuration for all experiments |
| `train_multiagent.py` | Main training script for multi-agent scenarios |
| `evaluate_multiagent.py` | Evaluation script for trained models |
| `algorithms.py` | Implementation of RL algorithms (PPO, DQN, etc.) |
| `results_{algoname}.json` | Training results and metrics for specified algorithm |
