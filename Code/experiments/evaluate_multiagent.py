# experiments/evaluate_multiagent.py
import os
import numpy as np
from Code.config import ENV_CONFIG
from algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
import torch

import warnings
warnings.filterwarnings("ignore")


def main(checkpoint_dir: str, max_steps: int = 20):

    # create factory and register environment with local config override
    local_config = ENV_CONFIG.copy()
    local_config.update({
        'use_gui': True,
        'num_seconds': max_steps * 5,  # duration based on delta_time=5
        'render_mode': 'human'
    })
    factory = AlgoConfigFactory(local_config)
    register_env(
        name="sumo_multi_agent",
        env_creator=lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    env = factory._create_env(local_config)

    # restore the algorithm from checkpoint
    algo = PPO.from_checkpoint(checkpoint_dir)
    # module = algo.get_module("shared_policy")
    # print(type(module))

    num_episodes = 5
    episode_rewards = {agent: [] for agent in env.possible_agents}

    # Get the trained module (policy) by ID
    module = algo.get_module("shared_policy")  

    def compute_actions(module, obs: dict):
        obs_array = np.array(list(obs.values()), dtype=np.float32)   # ensure ndarray
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # add batch dim

        actions_dict = {}

        # Forward pass through the policy (exploration mode = includes stochasticity)
        out = module.forward_inference({"obs": obs_tensor})

        action_dist_class = module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            out["action_dist_inputs"]
        )
        actions = action_dist.sample()[0].numpy()

        for i, agent_id in enumerate(env.possible_agents):
            actions_dict[agent_id] = actions[i]

        return actions_dict
        

    for ep in range(num_episodes):
        obs, _ = env.reset()
        rewards = {agent: 0 for agent in env.possible_agents}

        print(f"Starting episode {ep+1}... (watch SUMO-GUI)")

        while True:
            actions_dict = compute_actions(module, obs)

            obs, rew, terminated, truncated, _ = env.step(actions_dict)

            for agent_id, r in rew.items():
                rewards[agent_id] += r

            if all(terminated.values()) or all(truncated.values()):
                break

        # Store rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        print(f"Episode {ep+1} finished.")
        for agent_id in env.possible_agents:
            print(f"  Agent {agent_id}: Reward = {rewards[agent_id]}")

    env.close()

    print("\n=== Evaluation Summary ===")
    for agent_id, rewards in episode_rewards.items():
        print(f"Agent {agent_id}: mean reward = {np.mean(rewards)}")

if __name__ == "__main__":
    checkpoint_path = os.path.abspath("Code/outputs/checkpoints/ppo/200")
    main(checkpoint_path)
