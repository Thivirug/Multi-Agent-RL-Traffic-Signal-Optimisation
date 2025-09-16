import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from Code.config import ENV_CONFIG
from algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.sac import SAC
import torch
from tqdm import trange
import argparse

def main(checkpoint_dir: str, algo_name: str, max_steps: int = 20, num_episodes = 5):

    # create factory and register environment with local config override
    local_config = ENV_CONFIG.copy()
    local_config.update({
        'use_gui': True,
        'num_seconds': max_steps * local_config['delta_time'],  # duration based on delta_time
        'render_mode': 'human'
    })
    factory = AlgoConfigFactory(local_config)
    register_env(
        name="sumo_multi_agent",
        env_creator=lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    # create env with updated config for eval
    env = factory._create_env(local_config)

    # restore the required algorithm from checkpoint
    match algo_name:
        case 'ppo':
            algo = PPO.from_checkpoint(checkpoint_dir)
        case 'dqn':
            algo = DQN.from_checkpoint(checkpoint_dir)
        case 'sac':
            algo = SAC.from_checkpoint(checkpoint_dir)
        case _:
            raise ValueError("Algorithm can be ppo, dqn, or sac only !")

    episode_rewards = {agent: [] for agent in env.possible_agents}

    # Get the trained module (policy) by ID
    module = algo.get_module("shared_policy")  

    # inner helper function
    def compute_actions(module, obs: dict):
        """
            Get actions executed by each agent and create a dict to pass into env.step().
        """
        obs_array = np.array(list(obs.values()), dtype=np.float32)   # ensure ndarray
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # add batch dim

        actions_dict = {}

        # Forward pass through the policy (greedy in the sense it uses the learned policy)
        out = module.forward_inference({"obs": obs_tensor})

        # sample the best action
        action_dist_class = module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            out["action_dist_inputs"]
        )
        actions = action_dist.sample()[0].numpy()

        # create the actions dictionary
        for i, agent_id in enumerate(env.possible_agents):
            actions_dict[agent_id] = actions[i]

        return actions_dict
    
    # run eval loop for defined episodes
    for ep in trange(num_episodes):
        obs, _ = env.reset()
        
        # a dict to store rewards for each agent per episode
        rewards = {agent: 0 for agent in env.possible_agents}

        print(f"Starting episode {ep+1}... (watch SUMO-GUI)")

        while True:
            actions_dict = compute_actions(module, obs)

            # ! all these return values are dict (since multi agent)
            obs, rew, terminated, truncated, _ = env.step(actions_dict)

            for agent_id, r in rew.items():
                rewards[agent_id] += r

            # termination condition - if "ALL" agents are terminated or truncated
            if all(terminated.values()) or all(truncated.values()):
                break

        # store rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        print(f"\n ----- Episode {ep+1} finished ----- ")
        for agent_id in env.possible_agents:
            print(f"\n\tAgent {agent_id}: Reward = {rewards[agent_id]}")

        print() # for aesthetics 

    env.close()

    print("\n=== Evaluation Summary ===")
    for agent_id, rewards in episode_rewards.items():
        print(f"\nAgent {agent_id}: mean reward = {np.mean(rewards)}")

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained multi-agent algorithm checkpoint")
    p.add_argument("algo", help="Algorithm used for evaluation")
    p.add_argument("checkpoint", help="Path to checkpoint (directory)")
    p.add_argument("--max-steps", type=int, default=20, help="Max steps per episode (default: 20)")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    algoname = args.algo
    main(checkpoint_path, algoname, max_steps=args.max_steps, num_episodes=args.episodes)
