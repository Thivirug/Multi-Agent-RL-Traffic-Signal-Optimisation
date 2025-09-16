import os
import ray
import time
import numpy as np
from Code.config import ENV_CONFIG, PPO_hparams
from algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module.rl_module import RLModule
import torch

def main(checkpoint_dir: str, max_steps: int = 1000):

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

    obs, _ = env.reset()
    print(obs)
    obs_array = np.array(list(obs.values()), dtype=np.float32)   # ensure ndarray
    print(obs_array)
    obs_tensor = torch.from_numpy(obs_array).unsqueeze(0)  # add batch dim
    print(obs_tensor)

    # Forward pass through the policy (exploration mode = includes stochasticity)
    out = module.forward_inference({"obs": obs_tensor})
    
    action_dist_class = module.get_inference_action_dist_cls()
    action_dist = action_dist_class.from_logits(
        out["action_dist_inputs"]
    )
    actions = action_dist.sample()[0].numpy()

    print(actions)
    
    act_dct = {}
    for i, agent_id in enumerate(env.possible_agents):
        act_dct[agent_id] = actions[i]

    obs, rew, terminated, truncated, _ = env.step(act_dct)

    is_term = all(terminated.values())
    print(is_term)

    # print(done)
    # print(terminated)
    # print(truncated)
    # print(obs)


if __name__ == "__main__":
    checkpoint_path = os.path.abspath("Code/outputs/checkpoints/ppo/200")
    main(checkpoint_path, max_steps=2000)