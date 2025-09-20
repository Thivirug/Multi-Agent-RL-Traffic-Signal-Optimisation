import pandas as pd
import os
import matplotlib.pyplot as plt

from Code.experiments.algorithms import AlgoConfigFactory
from Code.config import ENV_CONFIG

import pettingzoo
import argparse

PLOTS_PARENT_PATH = os.path.abspath("Code/outputs/plots")
os.makedirs(PLOTS_PARENT_PATH, exist_ok=True)

# ! --------------------------------------- HELPER Funcs -------------------------------------------

def _get_results_df(json_file_path_template: str, algo_name: str) -> pd.DataFrame:
    """
        Return the pandas DataFrame of the given rewards .json file.

        Args:
            json_file_path_template (str): Path template to the json file with a placeholder for algo name.
            algo_name (str): Name of the algorithm used for evaluation (ppo/dqn/sac).

        Returns:
            DataFrame of the rewards json file.
    """
    try:
        json_file_path = json_file_path_template.format(algo_name=algo_name)
        return pd.read_json(json_file_path)
    except FileNotFoundError as e:
        print(f"File not found: {json_file_path}. Make sure the proper algo name is given (ppo/dqn/sac).")
        return pd.DataFrame() # return empty df to be consistent

def _get_single_agent_mean_ep_reward(agents_series: pd.Series, agent_id: str) -> list:
    """
        Return a list of the variation of mean episode reward of the given agent with iterations.

        Args:
            agents_series (pd.Series): Series of dicts mapping agent id to mean episode reward for that iteration.
            agent_id (str): The id of the agent whose rewards are to be extracted.

        Returns:
            List of mean episode rewards for the given agent.
    """
    list_of_rewards_of_agent = []

    # go through each row of the series
    for agents_dict in agents_series:
        assert isinstance(agents_dict, dict) # debug 

        list_of_rewards_of_agent.append(agents_dict[agent_id])
    
    return list_of_rewards_of_agent

def _get_agent_ids(env_config: dict) -> list:
    """
        Return a list of agent ids in the environment.

        Args:
            env_config (dict): The environment config to be passed into the factory.

        Returns:
            List of agent ids in the environment.
    """
    factory = AlgoConfigFactory(ENV_CONFIG)
    env: pettingzoo.utils.conversions.aec_to_parallel_wrapper = factory._create_env(ENV_CONFIG)

    return env.possible_agents    

def _create_rewards_dict_all_agents(agents_series: pd.Series, agent_ids: list) -> dict:
    """
        Return a dict mapping agent id to list of mean episode rewards for that agent.

        Args:
            agents_series (pd.Series): Series of dicts mapping agent id to mean episode reward for that iteration.
            agent_ids (list): List of agent ids in the environment.

        Returns:
            Dict mapping agent id to list of mean episode rewards for that agent.
    """
    rewards_dict = {}

    for agent_id in agent_ids:
        assert isinstance(agent_id, str) 

        rewards_list = _get_single_agent_mean_ep_reward(agents_series, agent_id)

        # convert id to int for ease
        agent_id = int(agent_id)

        rewards_dict[agent_id] = rewards_list

    return rewards_dict

# ! --------------------------------------- Main PLOTTING Funcs -------------------------------------------

def plot_mean_ep_reward(df: pd.DataFrame) -> None:
    """
        Plot the variation of mean episode reward with iterations and save it.

        Args:
            df (pd.DataFrame): DataFrame containing the results from the json file.
    """

    # check if df is empty
    if df.empty:
        print("DataFrame is empty. Cannot plot mean episode reward.")
        return
    
    plt.figure(figsize=(20,14))

    plt.plot(
        df['Iteration number'],
        df['Mean episode reward'],
        marker='o',
        linestyle='-',
        color='b'
    )

    plt.title('Mean Episode Reward vs Iterations', fontsize=30)
    plt.xlabel('Iterations', fontsize=24)
    plt.ylabel('Mean Episode Reward', fontsize=24)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(PLOTS_PARENT_PATH, 'mean_episode_reward.png'))
    print(f"Plot saved to {os.path.join(PLOTS_PARENT_PATH, 'mean_episode_reward.png')}")

def plot_all_agents_mean_ep_reward(df: pd.DataFrame, agent_ids: list) -> None:
    """
        Plot the variation of mean episode reward for all agents with iterations in a single fig and save it.

        Args:
            df (pd.DataFrame): DataFrame containing the results from the json file.
            agent_ids (list): List of agent ids in the environment.
    """

    # check if df is empty
    if df.empty:
        print("DataFrame is empty. Cannot plot mean episode reward.")
        return
    
    plt.figure(figsize=(20,14))

    # create plots for all agents

    # get the rewards dict mapping agent id to list of rewards
    rewards_dict = _create_rewards_dict_all_agents(df['Mean per agent reward'], agent_ids)
    # plot colours
    plot_colours = ['blue','green','red','black']

    for i, (id, r_list) in enumerate(rewards_dict.items()):
        plt.plot(
            df['Iteration number'],
            r_list,
            marker = 'o',
            linestyle = '-',
            color = plot_colours[i],
            label = f'{id}'
        )

    plt.title('Mean Episode Reward Per Agent vs Iterations', fontsize=30)
    plt.xlabel('Iterations', fontsize=24)
    plt.ylabel('Mean Episode Reward', fontsize=24)
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(PLOTS_PARENT_PATH, 'mean_reward_per_agent.png'))
    print(f"Plot saved to {os.path.join(PLOTS_PARENT_PATH, 'mean_reward_per_agent.png')}")

def _argparse() -> argparse.Namespace:
    """
        Parse given args from the terminal and return them to the program.
    """
    parser = argparse.ArgumentParser(description="Plot rewards from the json file for the given algorithm")
    parser.add_argument("algo", help="Algorithm used for evaluation (ppo/dqn/sac)")
    return parser.parse_args()

def main():
    # argparse
    args = _argparse()
    algo = args.algo
    df = _get_results_df(os.path.abspath(f"Code/outputs/results_{algo}.json"), algo)

    # get env agents ids
    ids = _get_agent_ids(ENV_CONFIG)

    # plot
    plot_mean_ep_reward(df)
    plot_all_agents_mean_ep_reward(df, ids)
    

if __name__ == '__main__':
    main()