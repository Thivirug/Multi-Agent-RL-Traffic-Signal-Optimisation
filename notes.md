# Required Knowledge for Implementing Multi-Agent Reinforcement Learning (MARL) for Traffic Signal Control (TSC)

Understanding the problem, setting up the environment, configuring the RL algorithm, and running/training the model. This is a *cooperative MARL* setup where each traffic light is an agent optimizing signals to reduce delays in a simulated network (e.g., a 4x4 grid). 

## 1. Core Problem and Concepts in TSC with MARL

- **Traffic Signal Control (TSC)**: TSC means adjusting traffic lights to minimize wait times, queues, or delays. In a city grid, fixed timers (e.g., green for 30 seconds) are inefficient; RL lets lights "learn" adaptive timing based on real-time traffic.
- **Multi-Agent RL (MARL)**: Single-agent RL is one decision-maker (e.g., a robot learning to walk). MARL has multiple agents (here, each intersection's light) interacting in a shared world. They learn together:
  - **Agents**: Traffic lights; each observes local traffic (e.g., car density on lanes) and acts (e.g., switch to next green phase).
  - **Environment**: The road network simulation where agents' actions affect each other (e.g., one light's delay causes backups elsewhere).
  - **Cooperation**: Agents share a goal (reduce total network delay) via a shared policy (same "brain" for all lights) to avoid chaos.
  - **Non-Stationarity**: As agents learn, the "world" changes for others, making training tricky—PPO handles this with stable updates.
  - **Rewards**: Negative for delays (e.g., 'diff-waiting-time' in sumo-rl: reward = change in total waiting time; better if less waiting).
  - **Observations/Actions**: Observations: Local lane stats (e.g., queue length). Actions: Discrete choices like "switch to phase 2" (green for certain directions), with yellow transitions.
- **Why MARL for TSC?**: Lights must coordinate (e.g., sync greens for smooth flow). MARL learns this implicitly through training.
- **Key Metric**: Aim to improve average waiting time or throughput; track via logs.

## 2. Simulation Environment: sumo-rl Basics

- **sumo-rl**: A library wrapping SUMO (Simulation of Urban Mobility), a free traffic simulator. It creates Gym/PettingZoo environments for RL.
  - **parallel_env**: For MARL; creates a multi-agent setup where agents act simultaneously every `delta_time` seconds (e.g., 5s).
  - **Config Keys**: `net_file` (road layout XML), `route_file` (vehicle flows), `num_seconds` (episode length, e.g., 3600s = 1 hour sim), `reward_fn` (e.g., 'diff-waiting-time'), `use_gui` (visualize, but off for training speed).
  - **Episode**: One full simulation run; ends after `num_seconds`. Agents get rewards per step.
- **PettingZoo Wrapper**: Makes sumo-rl compatible with MARL libs; agents act in parallel, environment steps forward together.
- **Customization**: If defaults don't work, tweak rewards (e.g., penalize queues more) by passing a custom function to `reward_fn`.

## 3. RL Algorithm: PPO in Ray RLlib

- **PPO (Proximal Policy Optimization)**: A reliable RL algorithm for this use case. It learns a policy (decision rule) by simulating episodes, calculating advantages (how good actions were), and updating gently to avoid big mistakes.
  - **Why PPO?**: Stable for MARL; uses clipping to prevent wild policy changes from non-stationarity.
  - **Shared Policy**: All agents use one policy (symmetric lights); map via `policy_mapping_fn=lambda agent_id: 'shared_policy'`.
- **RLlib**: Ray's RL library; handles MARL scaling.
  - **Config Structure**: Use `PPOConfig()` with method chaining (e.g., `.environment().training()`)—builds a config object like stacking Lego: set env, training params, multi-agent rules.
  - **Key Params**:
    - Environment: `.environment(env='sumo_multi_agent', env_config=your_dict)`.
    - Runners: `.env_runners(num_env_runners=2)` for parallel sims (speed up on multi-core CPU).
    - Training: `lr=0.0001` (learning speed), `train_batch_size=4000` (data per update), `entropy_coeff=0.01` (encourage exploration).
    - Multi-Agent: Define policies as dict (e.g., {'shared_policy': (None, None, None, {})}—inferred spaces).
    - Resources: `.resources(num_gpus=0)` (use 1 if available).
  - **Environment Registration**: `register_env('sumo_multi_agent', lambda config: ParallelPettingZooEnv(create_env(config)))`—tells RLlib how to make your sumo-rl env; lambda is a mini-function passing config to your env creator.
- **Training Loop**: Build algo with `config.build()`, then `algo.train()` in a loop (e.g., 100 iterations). Save checkpoints every 10 for resuming/eval.

* Implement PPO config in a class/file. Train on a small grid; monitor `episode_reward_mean` (higher = better cooperation).

## 4. Code Structure and Best Practices

- **Modular Files**: Separate configs (e.g., `config.py` for env/hparams), algorithms (e.g., class with `get_ppo_config()`), trainers (e.g., training loop with `ray.init()`).
  - `ray.init()`: In trainer script, before building algo; shuts down after.
  - Env Creator: Method to build `parallel_env`; lambda in register_env wraps it for RLlib.
- **Hyperparams**: Start with defaults; tune `lr`, `clip_param` if rewards stall.
- **Outputs**: Checkpoints in `.gitignore`; logs CSVs for metrics. Visualize with sumo-rl's plot script.
- **Debugging**: Set `use_gui=True` to watch sims; reduce `num_seconds` for quick tests.
- **Evaluation**: After training, `algo.evaluate()` on test routes; compare to fixed signals.
