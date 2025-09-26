# Training logic for multi-agents 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from Code.config import ENV_CONFIG, PPO_hparams, DQN_hparams, SAC_hparams, ARG_DICT
from algorithms import AlgoConfigFactory

from ray.tune.registry import register_env 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

import os
import re
from tqdm import trange

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json

def rename_logs(iter_n: int) -> None:
    """
        Rename the .csv log files to include iteration number
    """
    logs_dir = os.path.abspath("Code/outputs/logs")

    # define the pattern of strings of the filename
    pattern = re.compile(r"(logs_conn\d+_ep\d+)\.csv")

    for filename in os.listdir(logs_dir):
        # find matches
        match = pattern.match(filename)

        # make sure that previously renamed iteration logs are not renamed again
        if match and not f"_iter{iter_n}" in filename: # iter_n + 1 is not used to avoid a file naming error in sumo
            base = match.group(1)
            new_filename = f"{base}_iter{iter_n}.csv"
            os.rename(os.path.join(logs_dir, filename), os.path.join(logs_dir, new_filename))

def main() -> None:
    # # create and register env 
    # factory = AlgoConfigFactory(ENV_CONFIG)

    # # register our env
    # register_env(
    #     name = "sumo_multi_agent",
    #     env_creator = lambda config: ParallelPettingZooEnv(factory._create_env(config))
    # )
    
    # get algo config 
    print("\n ====== TRAINING START ======\n")
    print("Algorithm Options: ppo, dqn, sac")
    algo_name = input("Training algorithm: --> ")

    # update logs path with algo name
    ENV_CONFIG.update({
        'out_csv_name': os.path.abspath(f"Code/outputs/logs/{algo_name}/logs")
    })
    # create and register env 
    factory = AlgoConfigFactory(ENV_CONFIG)

    # get config
    match algo_name:
        case "ppo": 
            config = factory.get_ppo_config(PPO_hparams)
        case "dqn": 
            config = factory.get_dqn_config(DQN_hparams)
        case "sac":
            config = factory.get_sac_config(SAC_hparams)
        case _:
            raise ValueError("Algorithm can be ppo, dqn, or sac only !")
        
    # register our env
    register_env(
        name = "sumo_multi_agent",
        env_creator = lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    # use algorithm specific args
    n_iterations = ARG_DICT[algo_name]['n_iterations']
    checkpoint_freq = ARG_DICT[algo_name]['chkpoint_eval_freq']

    # build config
    algo = config.build()
    
    # list to store result dicts to be put into json
    results = []
    # make json file
    filename = f"results_{algo_name}.json"
    json_path = os.path.join(os.path.abspath("Code/outputs"), filename)

    # training loop
    for i in trange(n_iterations): 
        # run training loop
        algo.train()

        # rename logs
        rename_logs(i)

        # evaluate and checkpoint
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

    print("\n ====== TRAINING END ======\n")

    # close ray
    algo.stop()

main()