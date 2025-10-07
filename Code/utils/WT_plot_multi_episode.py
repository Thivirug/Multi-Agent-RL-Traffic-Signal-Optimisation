import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from typing import Dict, List

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

LOGS_DIR = os.path.abspath("Code/outputs/logs")
NEEDED_COL_NAMES = [
    "system_total_waiting_time",
    "system_mean_waiting_time",
    "1_accumulated_waiting_time",
    "2_accumulated_waiting_time",
    "5_accumulated_waiting_time",
    "6_accumulated_waiting_time",
    "agents_total_accumulated_waiting_time"
]

def find_iteration_log_files(algo: str, iteration: int) -> List[str]:
    """
        Find all log files belonging to a specific iteration.
        
        Args:
            algo (str): Algorithm name (e.g., 'ppo', 'dqn', 'sac')
            iteration (int): Iteration number to search for
        
        Returns:
            List of chronologically sorted log file paths for the specified iteration
    """

    # define the target suffix pattern
    log_pattern = os.path.join(LOGS_DIR, algo, f"*_iter{iteration}.csv")
    # get all matching log files
    log_files = glob.glob(log_pattern)
    
    # Sort files by episode number for consistent processing
    # make sure files are sorted in chronological order, not lexicographically
    def _extract_episode_num(filename):
        basename = os.path.basename(filename) 
        parts = basename.split('_')
        for part in parts:
            if part.startswith('ep'):
                return int(part.replace('ep', ''))
        return 0

    log_files.sort(key=_extract_episode_num) # apply key to all files and sort chronologically
    return log_files

def _load_dfs(log_files: List[str]) -> List[pd.DataFrame]:
    """
        Load dataframes from a list of log files.

        Args:
            log_files (List[str]): List of log file paths

        Returns:
            List of DataFrames corresponding to each log file

        Raises:
            ValueError: If no valid episode data is loaded
    """
    episode_dataframes = []
    
    print(f"Loading {len(log_files)} episode log files...")
    
    # create and append dataframes for each file (per episode)
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            # filter needed columns only
            df_filtered = df[NEEDED_COL_NAMES]
            episode_dataframes.append(df_filtered)
            print(f"Loaded: {os.path.basename(log_file)}")
        except Exception as e:
            print(f"Failed to load {os.path.basename(log_file)}: {e}")
            continue
    
    if not episode_dataframes: # empty list check
        raise ValueError("No valid episode data loaded!")

    return episode_dataframes

def load_and_aggregate_episodes(log_files: List[str]) -> Dict[str, pd.DataFrame]:
    """
        Load multiple episode log files and aggregate their metrics.
        
        Args:
            log_files (List[str]): List of log file paths
        
        Returns:
            Dict containing:
                - 'mean': DataFrame with mean values across episodes for interested metrics
                - 'std': DataFrame with standard deviations across episodes for interested metrics
                - 'min': DataFrame with minimum values across episodes for interested metrics
                - 'max': DataFrame with maximum values across episodes for interested metrics
                - 'episodes': List of individual episode DataFrames

        Raises:
            ValueError: If no valid episode data is loaded or no episodes with max length found
    """
    episode_dataframes = _load_dfs(log_files)
    
    # Find the max length across all episodes, then remove dfs that are shorter
    # (because some initial episodes of iterations are shorter especially in conn0)
    max_length = max(len(df) for df in episode_dataframes)
    episode_dataframes = [df for df in episode_dataframes if len(df) == max_length]
    if not episode_dataframes:
        raise ValueError("No episodes with the maximum length found!")
    
    print(f"Aggregating data from {len(episode_dataframes)} episodes of length {max_length}...")

    # Create 3D array: [episodes, time_steps, metrics]
    episodes_array = np.array([df.values for df in episode_dataframes])
    
    # Calculate aggregated statistics across episodes (axis 0) -> [time_steps, metrics]
    mean_data = np.mean(episodes_array, axis=0)
    std_data = np.std(episodes_array, axis=0)
    min_data = np.min(episodes_array, axis=0)
    max_data = np.max(episodes_array, axis=0)
    
    # Create DataFrames with proper column names and index
    index = range(episodes_array.shape[1])  # shape[1] is time_steps
    
    result = {
        'mean': pd.DataFrame(mean_data, columns=NEEDED_COL_NAMES, index=index),
        'std': pd.DataFrame(std_data, columns=NEEDED_COL_NAMES, index=index),
        'min': pd.DataFrame(min_data, columns=NEEDED_COL_NAMES, index=index),
        'max': pd.DataFrame(max_data, columns=NEEDED_COL_NAMES, index=index),
        'episodes': episode_dataframes,
        'n_episodes': len(episode_dataframes)
    }
    
    return result

def plot_time_series_individual_with_uncertainty(
    aggregated_data: Dict[str, pd.DataFrame], 
    algo_name: str, 
    iteration: int, 
    save_dir: str = "Code/outputs/plots"
) -> None:
    """
    Creates separate plots for each waiting time metric with uncertainty bands.
    
    Args:
        aggregated_data (Dict): Dictionary containing aggregated episode data
        algo_name (str): Algorithm name
        iteration (int): Iteration number
        save_dir (str): Directory to save the plots
    """
    print("Creating individual time series plots with uncertainty bands...")
    
    mean_df = aggregated_data['mean']
    std_df = aggregated_data['std']
    min_df = aggregated_data['min']
    max_df = aggregated_data['max']
    n_episodes = aggregated_data['n_episodes']
    
    for col in mean_df.columns: # iterate through each metric
        plt.figure(figsize=(14, 8))
        
        # Plot uncertainty band (mean ± std)
        plt.fill_between(
            mean_df.index, 
            mean_df[col] - std_df[col], 
            mean_df[col] + std_df[col],
            alpha=0.8, 
            color='lightcoral', 
            label='+/- Std Dev'
        )
        
        # Plot min/max envelope
        plt.fill_between(
            mean_df.index, 
            min_df[col], 
            max_df[col],
            alpha=0.3, 
            color='green', 
            label='Min-Max Range'
        )

        # Plot mean line
        plt.plot(
            mean_df.index,
            mean_df[col], 
            linewidth=3, 
            label=f'Mean ({n_episodes} episodes)', 
            color='darkblue'
        )
        
        plt.title(f"Traffic Signal Waiting Time: {col.replace('_', ' ').title()}\n"
                 f"Iteration {iteration} - {n_episodes} Episodes", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Simulation Time Step", fontsize=12)
        plt.ylabel("Waiting Time (seconds)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Statistics annotation for complete iteration
        mean_val = mean_df[col].mean()
        std_val = mean_df[col].std()
        
        plt.text(
            0.02, 
            0.98, 
            f'Iteration Mean: {mean_val:.2f}s\n'
            f'Iteration Std: {std_val:.2f}s\n'
            f'Episodes: {n_episodes}', 
            transform=plt.gca().transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Create output directory
        save_path = os.path.join(save_dir, algo_name, "waiting_time_multiEpisode", f"Iter_{iteration}", "individual")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_time_series_combined_with_uncertainty(
    aggregated_data: Dict[str, pd.DataFrame], 
    algo_name: str, 
    iteration: int, 
    save_dir: str = "Code/outputs/plots"
) -> None:
    """
        Shows multiple metrics on the same plot with uncertainty bands.
        
        Args:
            aggregated_data (Dict): Dictionary containing aggregated episode data
            algo_name (str): Algorithm name
            iteration (int): Iteration number
            save_dir (str): Directory to save the combined plot
    """
    print("Creating combined time series plot with uncertainty bands...")

    mean_df = aggregated_data['mean']
    std_df = aggregated_data['std']
    n_episodes = aggregated_data['n_episodes']
    
    # Create combined plot with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Traffic Signal Waiting Times - Iteration {iteration} ({n_episodes} episodes)', 
                 fontsize=16, fontweight='bold')
    
    # 1) System-level metrics
    axes[0,0].plot(mean_df.index, mean_df['system_total_waiting_time'], 
                   linewidth=1, label='Total Waiting Time', color='red')
    axes[0,0].fill_between(
        mean_df.index,
        mean_df['system_total_waiting_time'] - std_df['system_total_waiting_time'],
        mean_df['system_total_waiting_time'] + std_df['system_total_waiting_time'],
        alpha=0.8, color='lightcoral'
    )
    
    axes[0,0].plot(mean_df.index, mean_df['system_mean_waiting_time'], 
                   linewidth=1, label='Mean Waiting Time', color='blue')
    axes[0,0].fill_between(
        mean_df.index,
        mean_df['system_mean_waiting_time'] - std_df['system_mean_waiting_time'],
        mean_df['system_mean_waiting_time'] + std_df['system_mean_waiting_time'],
        alpha=0.8, color='lightblue'
    )
    
    axes[0,0].set_title('System-Level Waiting Times', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Waiting Time (s)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2) Individual intersection metrics
    intersection_cols = [col for col in mean_df.columns if col.startswith(('1_', '2_', '5_', '6_'))]
    colors = ['green', 'orange', 'purple', 'gray']
    
    for i, col in enumerate(intersection_cols):
        color = colors[i]
        axes[0,1].plot(mean_df.index, mean_df[col], linewidth=1, 
                      label=f'Intersection {col[0]}', color=color)
        axes[0,1].fill_between(
            mean_df.index,
            mean_df[col] - std_df[col],
            mean_df[col] + std_df[col],
            alpha=0.5, color=color
        )
    
    axes[0,1].set_title('Individual Intersection Waiting Times', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Accumulated Waiting Time (s)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3) Agents total accumulated
    axes[1,0].plot(mean_df.index, mean_df['agents_total_accumulated_waiting_time'], 
                   linewidth=3, label='Agents Total', color='green')
    axes[1,0].fill_between(
        mean_df.index,
        mean_df['agents_total_accumulated_waiting_time'] - std_df['agents_total_accumulated_waiting_time'],
        mean_df['agents_total_accumulated_waiting_time'] + std_df['agents_total_accumulated_waiting_time'],
        alpha=0.3, color='green'
    )
    
    axes[1,0].set_title('Total Agent Accumulated Waiting Time', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Waiting Time (s)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4) Episode-to-episode variability (coefficient of variation)
    cv_data = (std_df / mean_df) * 100  # Convert to percentage
    cv_data = cv_data.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
    
    for col in ['system_mean_waiting_time', 'agents_total_accumulated_waiting_time']:
        if col in cv_data.columns:
            axes[1,1].plot(mean_df.index, cv_data[col], linewidth=2, 
                          label=f'{col.replace("_", " ").replace("system ", "").replace("agents ", "").title()}')
    
    axes[1,1].set_title('Episode-to-Episode Variability (CV %)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Coefficient of Variation (%)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory
    save_path = os.path.join(save_dir, algo_name, "waiting_time_multiEpisode", f"Iter_{iteration}", "combined")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/combined_time_series.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_episode_comparison(
    aggregated_data: Dict[str, pd.DataFrame], 
    algo_name: str, 
    iteration: int, 
    metric: str,
    max_episodes_to_show: int = 10,
    save_dir: str = "Code/outputs/plots"
) -> None:
    """
        Plot individual episodes for comparison along with the mean.
        
        Args:
            aggregated_data (Dict): Dictionary containing aggregated episode data
            algo_name (str): Algorithm name
            iteration (int): Iteration number
            metric (str): Which metric to plot
            max_episodes_to_show (int): Maximum number of individual episodes to show
            save_dir (str): Directory to save the plots
    """
    print(f"Creating episode comparison plot for {metric}...")
    
    mean_df = aggregated_data['mean']
    episodes = aggregated_data['episodes']
    n_episodes = len(episodes)
    
    plt.figure(figsize=(16, 8))
    
    # Plot a subset/ complete set of individual episodes
    episodes_to_show = min(max_episodes_to_show, n_episodes)
    step = max(1, n_episodes // episodes_to_show)
    
    for i in range(0, n_episodes, step):
        if len([j for j in range(0, n_episodes, step)]) <= max_episodes_to_show:
            plt.plot(episodes[i].index, episodes[i][metric], 
                    alpha=0.3, linewidth=1, color='gray', 
                    label='Individual Episodes' if i == 0 else "")
    
    # Plot mean with thicker line
    plt.plot(mean_df.index, mean_df[metric], 
            linewidth=3, color='darkblue', label=f'Mean ({n_episodes} episodes)')
    
    plt.title(f"Episode Comparison: {metric.replace('_', ' ').title()}\n"
             f"Iteration {iteration} - Showing {episodes_to_show}/{n_episodes} episodes", 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Simulation Time Step", fontsize=12)
    plt.ylabel("Waiting Time (seconds)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Create output directory
    save_path = os.path.join(save_dir, algo_name, "waiting_time_multiEpisode", f"Iter_{iteration}", "episode_comparison")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{metric}_episode_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def _argparse() -> argparse.Namespace:
    """
        Parse given args from the terminal and return them to the program.
    """
    parser = argparse.ArgumentParser(description="Plot waiting time metrics from multiple episodes of a single iteration")
    parser.add_argument("algo", help="Algorithm used for evaluation (ppo/dqn/sac)")
    parser.add_argument("iteration", type=int, help="Iteration number to process (e.g., 480)")
    parser.add_argument("--max-episodes-comparison", type=int, default=10, 
                       help="Maximum number of episodes to show in comparison plot")
    return parser.parse_args()

def main():
    # Parse arguments
    args = _argparse()
    algo = args.algo
    iteration = args.iteration
    max_episodes_comparison = args.max_episodes_comparison

    print(f"Processing iteration {iteration} for algorithm {algo}")
    
    # Find all log files for the specified iteration
    log_files = find_iteration_log_files(algo, iteration)
    
    if not log_files:
        print(f"No log files found for iteration {iteration} in algorithm {algo}")
        print(f"Searched in: {os.path.join(LOGS_DIR, algo)}")
        return
    
    print(f"Found {len(log_files)} episode log files for iteration {iteration}")
    
    # Load and aggregate episode data
    try:
        aggregated_data = load_and_aggregate_episodes(log_files)
        print(f"Successfully aggregated data from {aggregated_data['n_episodes']} episodes")
        
        # Generate plots
        plot_time_series_individual_with_uncertainty(aggregated_data, algo, iteration)
        plot_time_series_combined_with_uncertainty(aggregated_data, algo, iteration)
        
        # Create episode comparison plots for key metrics
        key_metrics = ['system_mean_waiting_time', 'agents_total_accumulated_waiting_time']
        for metric in key_metrics:
            if metric in NEEDED_COL_NAMES:
                plot_episode_comparison(aggregated_data, algo, iteration, metric, max_episodes_comparison)
        
        print(f"All plots generated successfully!")
        print(f"Output directory: Code/outputs/plots/{algo}/waiting_time_multiEpisode/Iter_{iteration}/")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return

if __name__ == "__main__":
    main()