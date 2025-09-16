# Training logic for multi-agents 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from Code.config import ENV_CONFIG, PPO_hparams, DQN_hparams, n_iterations, checkpoint_freq
from algorithms import AlgoConfigFactory

from ray.tune.registry import register_env 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

import os
import re
from tqdm import trange

import json

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
    
    # list to store result dicts to be put into json
    results = []
    json_path = os.path.abspath("Code/outputs/results.json")

    # training loop
    for i in trange(n_iterations): # 1 iteration =  "train_batch_size_per_learner" timesteps # ! use tqdm
        algo.train()

        # rename logs
        rename_logs(i)

        if (i + 1) % checkpoint_freq == 0:
            # evaluate algorithm till now
            result = algo.evaluate()

            # print iteration progress message
            mean_episode_reward = result['env_runners'].get('episode_return_mean', {}) 
            print(f"\n\t -- Iteration {i+1} --- \n\tMean Episode Reward: {mean_episode_reward:.5f}")

            # print per agent reward
            per_agent_mean = result['env_runners'].get('agent_episode_returns_mean', {})
            print(f"\tPer Agent Mean Episode Reward: {per_agent_mean}")

            # checkpointing
            chkpoint_dir = os.path.abspath(f"Code/outputs/checkpoints/{algo_name}/{i+1}")
            os.makedirs(chkpoint_dir, exist_ok=True)
            chkpoint_path = algo.save(chkpoint_dir)
            print(f"\nCheckpoint saved to {chkpoint_path}\n")

            # appending to results
            result = {
                "Iteration number": i+1,
                "Mean episode reward": mean_episode_reward,
                "Mean per agent reward": per_agent_mean
            }
            results.append(result)

    # dump to json
    print(f"\n\t Dumping Results to {json_path}...\n")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # close ray
    algo.stop()
    # ray.shutdown()

main()