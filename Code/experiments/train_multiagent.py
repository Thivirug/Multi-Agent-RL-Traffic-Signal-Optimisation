# Training logic for multi-agents 
import sys
import os
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from Code.config import ENV_CONFIG, PPO_hparams, DQN_hparams, n_iterations, checkpoint_freq
from algorithms import AlgoConfigFactory

from ray.tune.registry import register_env 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray import tune

import ray
import os
import numpy as np

import re

def rename_logs(iter_n: int):
    """
        Rename the .csv log files to include iteration number
    """
    logs_dir = os.path.abspath("Code/outputs/logs")
    pattern = re.compile(r"(logs_conn\d+_ep\d+)\.csv")

    for filename in os.listdir(logs_dir):
        match = pattern.match(filename)
        if match and not f"_iter{iter_n}" in filename: # iter_n + 1 is not used to avoid a file naming error in sumo
            base = match.group(1)
            new_filename = f"{base}_iter{iter_n}.csv"
            os.rename(os.path.join(logs_dir, filename), os.path.join(logs_dir, new_filename))

def main():
    # # init ray
    # ray.init() 
    
    # create and register env 
    factory = AlgoConfigFactory(ENV_CONFIG)

    register_env(
        name = "sumo_multi_agent",
        env_creator = lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )
    
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
    for i in range(n_iterations): # 1 iteration =  "train_batch_size_per_learner" timesteps # ! use tqdm
        algo.train()

        # rename logs
        rename_logs(i)
    
        # result = algo.evaluate()
        
        # # ! MIGHT NEED TO SAVE TO JSON
        # print(f"\n\t -- Iteration {i+1} --- \n\tMean Episode Reward : {result['env_runners']['episode_return_mean']:.5f}")
        # # pprint.pprint(f"Per Agent Mean Episode Reward : {result['env_runners']['agent_episode_returns_mean']:.5f}")

        # # checkpoint every freq-th iter
        # if i+1 % checkpoint_freq == 0:
        #     chkpoint_dir = os.path.abspath(f"Code/outputs/checkpoints/{algo_name}/{i+1}") 
        #     os.makedirs(chkpoint_dir, exist_ok=True)
        #     chkpoint_path = algo.save_to_path(chkpoint_dir)
        #     print(f"/nCheckpoint saved to {chkpoint_path}/n")

        if (i + 1) % checkpoint_freq == 0:
            result = algo.evaluate()
            print(f"\n\t -- Iteration {i+1} --- \n\tMean Episode Reward: {result['env_runners']['episode_return_mean']:.5f}")
            per_agent_mean = result['env_runners'].get('agent_episode_returns_mean', {})
            print(f"\tPer Agent Mean Episode Reward: {per_agent_mean}")
            chkpoint_dir = os.path.abspath(f"Code/outputs/checkpoints/{algo_name}/{i+1}")
            os.makedirs(chkpoint_dir, exist_ok=True)
            chkpoint_path = algo.save(chkpoint_dir)
            print(f"\nCheckpoint saved to {chkpoint_path}\n")

    # close ray
    algo.stop()
    # ray.shutdown()

main()