import os
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'  

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.getLogger('pyvirtualdisplay').setLevel(logging.ERROR)

import numpy as np
from Code.config import ENV_CONFIG
from Code.experiments.algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.sac import SAC
import torch
from tqdm import trange
import argparse
from pyvirtualdisplay import Display  
import time
import cv2
import pettingzoo
import abc

def _get_env(max_steps: int, env_config: dict) -> pettingzoo.utils.conversions.aec_to_parallel_wrapper:
    """
        Create a parallel env in SUMO with the configs required for evaluation and recording.

        Args:
            max_steps (int): Max steps per episode. 
            env_config (dict): Base environment configuration.

        Returns:
            The updated environment for eval and recording.
    """
    # create factory and register environment with local config override
    local_config = env_config.copy()
    local_config.update({
        'use_gui': False,  # headless for rgb_array
        'num_seconds': max_steps * local_config['delta_time'],
        'render_mode': 'rgb_array'
    })
    factory = AlgoConfigFactory(local_config)
    register_env(
        name="sumo_multi_agent",
        env_creator=lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    # create env for evaluation
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
        obs_array = np.array(list(obs.values()), dtype=np.float32)   # Ensure ndarray
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # Add batch dim

        actions_dict = {}

        # forward pass through the policy - get action logits
        out = module.forward_inference({"obs": obs_tensor})

        # ! actions selection 

        # ! For PPO & SAC - Outputs logits for each possible action for each agent
        if out.get("action_dist_inputs") is not None:
            # retrieve the class for the action distribution used during inference
            action_dist_class: abc.ABCMeta = module.get_inference_action_dist_cls()

            # get probability distribution over actions
            action_dist: TorchCategorical = action_dist_class.from_logits(out["action_dist_inputs"])

            # get the actions - greedy (deterministic) 
            actions: np.ndarray = action_dist.to_deterministic().sample()[0].numpy()

        # ! For DQN - Outputs the action indices directly for each agent
        if out.get("actions") is not None:
            action_indices: torch.Tensor = out["actions"] # shape (1, num_agents)
            actions = action_indices[0].numpy()  # shape (num_agents,)

        # Create the actions dictionary
        for i, agent_id in enumerate(env.possible_agents):
            actions_dict[agent_id] = actions[i]

        return actions_dict

def _init_video_rec(video_dir: str, checkpoint_dir: str, algo_name: str, ep: int, max_steps: int, env: pettingzoo.utils.conversions.aec_to_parallel_wrapper) -> tuple[cv2.VideoWriter, np.ndarray]:
    """
        Initialise video recording setup for given episode.

        Args:
            video_dir (str): Directory to save videos.
            algo_name (str): Algorithm name for filename.
            ep (int): Current episode number.
            max_steps (int): Max steps per episode.
            env (pettingzoo.utils.conversions.aec_to_parallel_wrapper): The environment instance.

        Returns:
            A tuple of (cv2.VideoWriter object, initial frame ndarray).
    """
    # Initialise video writer with OpenCV
    video_filename = os.path.join(
        video_dir, 
        f'algorithm_{algo_name}_episode#_{ep+1}_iteration#_{_get_chkpoint_iteration(checkpoint_dir)}_max_steps_{max_steps}.mp4'
    )

    frame = env.render()  # Get initial frame to determine dims
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    return (
         cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height)),
         frame # return initial frame for writing
    )  

def _get_chkpoint_iteration(checkpoint_dir: str) -> int:
    """
        Extract the training iteration number from the checkpoint directory name.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.

        Returns:
            The iteration number as an integer.

        Raises:
            ValueError: If the iteration number cannot be extracted.
    """
    # this extraction gives exactly the iteration number
    try:
        base_name = os.path.basename(checkpoint_dir)
        return int(base_name)
    except Exception as e:
        raise ValueError(f"Could not extract iteration number from checkpoint directory name '{checkpoint_dir}'. Ensure it ends with the iteration number.") from e
    
def _parse_args() -> argparse.Namespace:
    """
        Parse given args from the terminal and return them to the program.
    """
    p = argparse.ArgumentParser(description="Evaluate trained multi-agent algorithm checkpoint")
    # mandatory args
    p.add_argument("algo", help="Algorithm used for evaluation")
    p.add_argument("checkpoint", help="Path to checkpoint (directory)")

    # optional args
    p.add_argument("--max-steps", type=int, default=20, help="Max steps per episode (default: 20)")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    return p.parse_args()

def main(checkpoint_dir: str, algo_name: str, max_steps: int = 20, num_episodes: int = 5) -> None:
    """
        Run evaluation of a trained multi-agent RL algorithm checkpoint and record videos.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            algo_name (str): Algorithm name (e.g., 'ppo', 'dqn', 'sac').
            max_steps (int): Max steps per episode (default: 20).
            num_episodes (int): Number of episodes to run (default: 5).
    """
    # get eval env
    env = _get_env(max_steps, ENV_CONFIG)

    # Restore the required algorithm from checkpoint
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

    # Setup virtual display
    display = None
    try:
        display = Display(visible=0, size=(1280, 720))
        display.start()
        print("Virtual display started successfully.")
        time.sleep(2)  # Allow display to initialize
    except Exception as e:
        print(f"Virtual display failed to start: {e}.")
        display = None

    # Directory to save videos
    video_dir = os.path.join(os.path.abspath("Code/outputs/recordings"), f"{algo_name}")
    os.makedirs(video_dir, exist_ok=True)

    # Run eval loop for defined episodes
    for ep in trange(num_episodes):
        obs, _ = env.reset()
        time.sleep(5)  # Allow TraCI to stabilize

        rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Starting episode {ep+1}... ")

        out, initial_frame = _init_video_rec(video_dir, checkpoint_dir, algo_name, ep, max_steps, env)
        out.write(cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR))  # Write initial frame

        while True:
            # get the actions for all agents
            actions_dict = _compute_actions(module, obs, env)

            obs, rew, terminated, truncated, _ = env.step(actions_dict)

            frame = env.render() # Get frame after step
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write frame

            # accumulate rewards for each agent
            for agent_id, r in rew.items():
                rewards[agent_id] += r

            # check for episode termination (all agents must be done)
            if all(terminated.values()) or all(truncated.values()):
                break

        # release resources
        out.release()
        print(f"Video saved for episode {ep+1}... ")

        # store rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        print(f"\n ----- Episode {ep+1} finished ----- ")
        for agent_id in env.possible_agents:
            print(f"\n\tAgent {agent_id}: Reward = {rewards[agent_id]}")

        print()  # For aesthetics

    env.close()

    # stop virtual display
    if display:
        display.stop()

    print("\n=== Evaluation Summary ===")
    for agent_id, rewards in episode_rewards.items():
        print(f"\nAgent {agent_id}:")
        print(f"\tmean reward = {np.mean(rewards)}")
        print(f"\tstd reward = {np.std(rewards)}")

if __name__ == "__main__":
    args = _parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    main(checkpoint_path, args.algo, args.max_steps, args.episodes)