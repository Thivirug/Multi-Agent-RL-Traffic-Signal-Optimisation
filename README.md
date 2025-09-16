
#### Tentative Project Directory Structure

```toml
# Project Root
[project_root]

[project_root.Code]
config = "config.py"

# Experiments
[project_root.Code.experiments]
algorithms = "algorithms.py"
evaluate_multiagent = "evaluate_multiagent.py"
todo = "todo.txt"
train_multiagent = "train_multiagent.py"

# Initial Tests
[project_root.Code.initial_testing]
test_gpu = "test_gpu.py"
test_gpu_mac = "test_gpu_mac.py"
test_install = "test_install.py"
test_mac = "test_mac.py"

# Outputs
[project_root.Code.outputs]
results_ppo = "results_ppo.json"
results_dqn = "results_dqn.json"
results_sac = "results_sac.json"
todo = "todo.txt"

[project_root.Code.outputs.checkpoints.ppo]
[project_root.Code.outputs.checkpoints.dqn]
[project_root.Code.outputs.checkpoints.sac]
logs = "[many .csv log files]"

# Utilities
[project_root.Code.utils]
record_eval = "record_eval.py"
rewards_plot = "rewards_plot.py"
waiting_time_plot = "waiting_time_plot.py"

# Documentation & Dependencies
[project_root.docs]
MARL_notes = "MARL_notes.md"
README = "README.md"
mac_requirements = "mac_requirements.txt"
notes = "notes.md"
requirements = "requirements.txt"

# Source Code
[project_root.src]
sumo-rl = "sumo-rl/"
```
