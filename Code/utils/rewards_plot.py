import pandas as pd
import os

PLOTS_PARENT_PATH = os.path.abspath("Code/outputs/plots")
os.makedirs(PLOTS_PARENT_PATH, exist_ok=True)

def get_results_df(json_file_path: str) -> pd.DataFrame:
    """
        Return the pandas DataFrame of the given rewards .json file.
    """
    pass

def plot_mean_ep_reward(df: pd.DataFrame) -> None:
    """
        Plot the variation of mean episode reward with iterations and save it.
    """
    pass

def plot_single_agent_mean_ep_reward(df: pd.DataFrame, agent_id: int) -> None:
    """
        Plot the variation of mean episode reward of the given agent with iterations.
    """
    pass

def plot_all_agents_mean_ep_reward(df: pd.DataFrame, agent_ids: list) -> None:
    """
        Plot the variation of mean episode reward for all agents with iterations in a single fig and save it.
    """
    pass

def _argparse():
    pass

def main():
    pass

if __name__ == 'main':
    # args
    main()