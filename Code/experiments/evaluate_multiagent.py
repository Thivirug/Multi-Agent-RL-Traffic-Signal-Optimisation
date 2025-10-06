import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from Code.config import ENV_CONFIG
from algorithms import AlgoConfigFactory

from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.appo import APPO

import torch
from tqdm import trange
import argparse
import pettingzoo

def _get_env(max_steps: int, env_config: dict) -> pettingzoo.utils.conversions.aec_to_parallel_wrapper:
    """
        Create a parallel env in SUMO with the configs required for evaluation.

        Args:
            max_steps (int): Max steps per episode. 
            env_config (dict): Base environment configuration.

        Returns:
            The updated environment for evaluation.
    """
    # create factory and register environment with local config override
    local_config = env_config.copy()
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
    return factory._create_env(local_config)

def _compute_actions(module: MultiRLModule, obs: dict, env: pettingzoo.utils.conversions.aec_to_parallel_wrapper) -> dict:
    """
        Get actions executed by each agent and create a dict to pass into env.step().

        Args:
            module (MultiRLModule): The trained policy module.
            obs (dict): Current observations from all agents.
            env (pettingzoo.utils.conversions.aec_to_parallel_wrapper): The environment instance.

        Returns:
            Actions for each agent.
    """
    obs_array = np.array(list(obs.values()), dtype=np.float32)   # ensure ndarray
    obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # add batch dim

    actions_dict = {}

    # Forward pass through the policy (greedy in the sense it uses the learned policy)
    out = module.forward_inference({"obs": obs_tensor})

    # ! For PPO & SAC- Outputs logits for each possible action for each agent
    if out.get("action_dist_inputs") is not None:
        # sample the best action
        action_dist_class = module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            out["action_dist_inputs"]
        )
        actions = action_dist.sample()[0].numpy() # ! using sampling instead of deterministic evaluation

    # ! For DQN - Outputs the action indices directly for each agent
    if out.get("actions") is not None:
        action_indices = out["actions"] # shape (1, num_agents)
        actions = action_indices[0].numpy()  # shape (num_agents,)

    # create the actions dictionary
    for i, agent_id in enumerate(env.possible_agents):
        actions_dict[agent_id] = actions[i]

    return actions_dict

def _parse_args() -> argparse.Namespace:
    """
        Parse given args from the terminal and return them to the program.
    """
    p = argparse.ArgumentParser(description="Evaluate trained multi-agent algorithm checkpoint")
    p.add_argument("algo", help="Algorithm used for evaluation")
    p.add_argument("checkpoint", help="Path to checkpoint (directory)")
    p.add_argument("--max-steps", type=int, default=20, help="Max steps per episode (default: 20)")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    return p.parse_args()

def main(checkpoint_dir: str, algo_name: str, max_steps: int = 20, num_episodes: int = 5) -> None:
    """
        Evaluate a trained multi-agent RL algorithm in the SUMO environment and view results.

        Args:
            checkpoint_dir (str): Path to the directory containing the trained model checkpoint.
            algo_name (str): The name of the algorithm used for training ('ppo', 'dqn', or 'sac').
            max_steps (int, optional): Maximum steps per episode. Defaults to 20.
            num_episodes (int, optional): Number of episodes to run for evaluation. Defaults to 5.
    """
    # get eval env
    env = _get_env(max_steps, ENV_CONFIG)

    # restore the required algorithm from checkpoint
    match algo_name:
        case 'ppo':
            algo = PPO.from_checkpoint(checkpoint_dir)
        case 'dqn':
            algo = DQN.from_checkpoint(checkpoint_dir)
        case 'sac':
            algo = SAC.from_checkpoint(checkpoint_dir)
        case 'appo':
            algo = APPO.from_checkpoint(checkpoint_dir)
        case _:
            raise ValueError("Algorithm can be ppo, dqn, or sac only !")

    episode_rewards = {agent: [] for agent in env.possible_agents}

    # Get the trained module (policy) by ID
    module = algo.get_module("shared_policy")  

    print(type(module))
    
    # run eval loop for defined episodes
    for ep in trange(num_episodes):
        obs, _ = env.reset()
        
        # a dict to store rewards for each agent per episode
        rewards = {agent: 0 for agent in env.possible_agents}

        print(f"Starting episode {ep+1}... (watch SUMO-GUI)")

        while True:
            # get the actions for all agents
            actions_dict = _compute_actions(module, obs, env)

            # ! all these return values are dict (since multi agent)
            obs, rew, terminated, truncated, _ = env.step(actions_dict)

            # accumulate rewards for each agent
            for agent_id, r in rew.items():
                rewards[agent_id] += r

            # check for episode termination (all agents must be done)
            if all(terminated.values()) or all(truncated.values()):
                break

        # store rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        print(f"\n ----- Episode {ep+1} finished ----- ")
        for agent_id in env.possible_agents:
            print(f"\n\tAgent {agent_id}: Reward = {rewards[agent_id]}")

        print()  # For aesthetics

    env.close()

    print("\n=== Evaluation Summary ===")
    for agent_id, rewards in episode_rewards.items():
        print(f"\nAgent {agent_id}:")
        print(f"\tmean reward = {np.mean(rewards)}")
        print(f"\tstd reward = {np.std(rewards)}")

if __name__ == "__main__":
    args = _parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    main(checkpoint_path, args.algo, args.max_steps, args.episodes)
