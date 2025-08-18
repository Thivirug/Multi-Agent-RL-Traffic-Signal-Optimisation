# Training logic for multi-agents 

from Code.config import ENV_CONFIG, PPO_hparams, DQN_hparams, n_iterations, checkpoint_freq
from algorithms import AlgoConfigFactory

import ray
import os

def main():
    # init ray
    ray.init(ignore_reinit_error = True) # allow re-initialisation

    # create and register env 
    factory = AlgoConfigFactory(ENV_CONFIG)
    
    # get algo config 
    algo_name = input("Training algorithm: --> ")

    # get config
    match algo_name:
        case "ppo": 
            config = factory.get_ppo_config(PPO_hparams)
        case "dqn": 
            config = factory.get_dqn_config(DQN_hparams)
        # other one

    # build config
    algo = config.build()

    # training loop
    for i in range(n_iterations):
        result = algo.train()
        print(f"Iteration {i}: Reward = {result['episode_reward_mean']:.5f}")

        # checkpoint every freq-th iter
        if i % checkpoint_freq == 0:
            chkpoint_dir = f"Code/outputs/checkpoints/{algo_name}"
            os.makedirs(chkpoint_dir, exist_ok=True)
            chkpoint_path = algo.save_to_path()
            print(f"Checkpoint saved to {chkpoint_path}")

    # close ray
    ray.shutdown()


if __name__ == 'main':
    main()