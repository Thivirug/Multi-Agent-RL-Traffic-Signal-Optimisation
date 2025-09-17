import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
from Code.config import ENV_CONFIG
from Code.experiments.algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.sac import SAC
import torch
from tqdm import trange
import argparse
import subprocess
import time
import ray

def main(checkpoint_dir: str, algo_name: str, max_steps: int = 20, num_episodes: int = 5) -> None:
    # Initialize Ray with single process to avoid conflicts
    ray.init(num_cpus=1, ignore_reinit_error=True)

    # Create factory and register environment with local config override
    local_config = ENV_CONFIG.copy()
    local_config.update({
        'use_gui': True,  # Enable SUMO GUI for recording
        'num_seconds': max_steps * local_config['delta_time'],  # Duration based on delta_time
        'additional_sumo_cmd': '--remote-port 8813 --step-length 5.0'  # Explicit port and step length
    })
    factory = AlgoConfigFactory(local_config)
    register_env(
        name="sumo_multi_agent",
        env_creator=lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    # Create env with updated config for eval
    env = factory._create_env(local_config)

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

    # Inner helper function
    def compute_actions(module, obs: dict) -> dict:
        """
        Get actions executed by each agent and create a dict to pass into env.step().
        """
        obs_array = np.array(list(obs.values()), dtype=np.float32)   # Ensure ndarray
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # Add batch dim

        actions_dict = {}

        # Forward pass through the policy
        out = module.forward_inference({"obs": obs_tensor})

        # Sample the best action
        action_dist_class = module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(out["action_dist_inputs"])
        actions = action_dist.sample()[0].numpy()

        # Create the actions dictionary
        for i, agent_id in enumerate(env.possible_agents):
            actions_dict[agent_id] = actions[i]

        return actions_dict
    
    # Run eval loop for defined episodes
    for ep in trange(num_episodes):
        obs, _ = env.reset()
        time.sleep(5)  # Allow TraCI to stabilize

        rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Starting episode {ep+1}... (watch SUMO-GUI)")

        # Start ffmpeg recording (automated)
        video_dir = os.path.join(os.path.abspath("Code/outputs"), 'recordings')
        os.makedirs(video_dir, exist_ok=True)
        video_filename = os.path.join(video_dir, f'{algo_name}_episode_{ep+1}.mp4')
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'x11grab',
            '-video_size', '1280x720',  # Adjust to your SUMO window size
            '-i', ':0.0',  # Display :0 (default GUI display)
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-y',  # Overwrite output files
            video_filename
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while True:
            actions_dict = compute_actions(module, obs)
            obs, rew, terminated, truncated, _ = env.step(actions_dict)

            for agent_id, r in rew.items():
                rewards[agent_id] += r

            if all(terminated.values()) or all(truncated.values()):
                break

        # Stop ffmpeg recording
        ffmpeg_process.terminate()
        time.sleep(1)  # Allow ffmpeg to finalize

        # Store rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(rewards[agent_id])

        print(f"\n ----- Episode {ep+1} finished ----- ")
        for agent_id in env.possible_agents:
            print(f"\n\tAgent {agent_id}: Reward = {rewards[agent_id]}")

        print()  # For aesthetics
        print(f"Video saved for episode {ep+1} at {video_filename}")

    env.close()
    ray.shutdown()

    print("\n=== Evaluation Summary ===")
    for agent_id, rewards in episode_rewards.items():
        print(f"\nAgent {agent_id}: mean reward = {np.mean(rewards)}")

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

if __name__ == "__main__":
    args = _parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    algoname = args.algo
    main(checkpoint_path, algoname, max_steps=args.max_steps, num_episodes=args.episodes)