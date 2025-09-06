# Training logic for multi-agents 

from Code.config import ENV_CONFIG, PPO_hparams, DQN_hparams, n_iterations, checkpoint_freq
from algorithms import AlgoConfigFactory

from ray.tune.registry import register_env 
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray import tune

import ray
import os
import numpy as np

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

    # # training loop
    # for i in range(n_iterations): # 1 iteration =  "train_batch_size_per_learner" timesteps
    #     results = algo.train()
    #     # print(f"Iteration {i}: Reward = {result['policy_reward_mean']:.5f}")
    #     # print(type(result))
    #     mean_return = results["env_runners"].get(
    #                 "episode_return_mean", np.nan
    #             )
    #     print(f"\titer={i} R={mean_return}\n")

    #     # checkpoint every freq-th iter
    #     if i % checkpoint_freq == 0:
    #         chkpoint_dir = os.path.abspath(f"Code/outputs/checkpoints/{algo_name}/{i}") 
    #         os.makedirs(chkpoint_dir, exist_ok=True)
    #         chkpoint_path = algo.save_to_path(chkpoint_dir)
    #         print(f"Checkpoint saved to {chkpoint_path}")


    # # ! Use Tuner

    # # Stop when we've either reached 100 training iterations or reward=300
    # stopping_criteria = {"training_iteration": 5}
    
    # tuner = tune.Tuner(
    #     "PPO",
    #     tune_config=tune.TuneConfig(
    #         metric="episode_return_mean",
    #         mode="max",
    #     ),
    #     param_space=config,
    #     run_config=tune.RunConfig(stop=stopping_criteria),
    # )
    # results = tuner.fit()
    
    import pprint
    # training loop
    for i in range(1): # 1 iteration =  "train_batch_size_per_learner" timesteps
        results = algo.train()
        # # print(f"Iteration {i}: Reward = {result['policy_reward_mean']:.5f}")
        # # print(type(result))
        # mean_return = results["env_runners"].get(
        #             "episode_return_mean", np.nan
        #         )
        # print(f"\titer={i} R={mean_return}\n")
    
    pprint.pprint(algo.evaluate())

    # best_result = results.get_best_result()

    # pprint.pprint(best_result)

    # print("\nBest performing trial's final reported metrics:\n")

    # metrics_to_print = [
    #     "episode_reward_mean",
    #     "episode_reward_max",
    #     "episode_reward_min",
    #     "episode_len_mean",
    # ]
    # pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

    # close ray
    algo.stop()
    # ray.shutdown()

main()