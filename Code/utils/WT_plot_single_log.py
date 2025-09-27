import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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

# ! Plot 1: Individual Time Series Plots
def plot_time_series_individual(df: pd.DataFrame, algo_name: str, save_dir: str = "Code/outputs/plots") -> None:
    """
        Creates separate plots for each waiting time metric over simulation steps.

        Args:
            df (pd.DataFrame): DataFrame containing waiting time metrics.
            save_dir (str): Directory to save the plots.
    """
    print("Creating individual time series plots...")
    
    for col in df.columns:
        plt.figure(figsize=(12, 6))
        
        sns.lineplot(data=df, x=df.index, y=col, linewidth=2.5, alpha=0.8)
        
        plt.title(f"Traffic Signal Waiting Time: {col.replace('_', ' ').title()}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Simulation Time Step", fontsize=12)
        plt.ylabel("Waiting Time (seconds)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # statistics annotation
        mean_val = df[col].mean()
        std_val = df[col].std()
        max_val = df[col].max()
        min_val = df[col].min()
        plt.text(
            0.02, 
            0.98, 
            f'Mean: {mean_val:.2f}s\nStd: {std_val:.2f}s\nMax: {max_val:.2f}s\nMin: {min_val:.2f}s', 
            transform = plt.gca().transAxes, 
            verticalalignment = 'top',
            bbox = dict(boxstyle = 'round', 
                        facecolor = 'wheat', 
                        alpha = 0.8
                )
        )
        
        plt.tight_layout()
        
        # create output directory
        SAVE_DIR = os.path.join(save_dir, algo_name, "waiting_time", "individual")
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(f"{SAVE_DIR}/{col}.png", dpi=300, bbox_inches='tight')
        plt.show()

# ! Plot 2: Combined Time Series Plot
def plot_time_series_combined(df: pd.DataFrame, algo_name: str, save_dir: str = "Code/outputs/plots") -> None:
    """
        Shows multiple metrics on the same plot for comparison.
        (Useful for understanding relationships between different waiting time metrics.)

        Args:
            df (pd.DataFrame): DataFrame containing waiting time metrics.
            save_dir (str): Directory to save the combined plot.
    """
    print("Creating combined time series plot...")

    # create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # combined plot with subplots
    _, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1) System-level metrics
    axes[0,0].plot(df.index, df['system_total_waiting_time'], 
                   linewidth=2.5, label='Total Waiting Time', color='red')
    axes[0,0].plot(df.index, df['system_mean_waiting_time'], 
                   linewidth=2.5, label='Mean Waiting Time', color='blue')
    axes[0,0].set_title('System-Level Waiting Times', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Waiting Time (s)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2) Individual intersection metrics
    intersection_cols = [col for col in df.columns if col.startswith(('1_', '2_', '5_', '6_'))]
    for i, col in enumerate(intersection_cols):
        axes[0,1].plot(df.index, df[col], linewidth=2, label=f'Intersection {col[0]}', alpha=0.8)
    axes[0,1].set_title('Individual Intersection Waiting Times', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Accumulated Waiting Time (s)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3) Agents total accumulated
    axes[1,0].plot(df.index, df['agents_total_accumulated_waiting_time'], 
                   linewidth=2.5, label='Agents Total', color='green')
    axes[1,0].set_title('Total Agent Accumulated Waiting Time', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Waiting Time (s)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4) Moving average of system mean waiting time
    window_size = max(10, len(df) // 20)  # Adaptive window size
    moving_avg = df['system_mean_waiting_time'].rolling(window=window_size).mean()
    axes[1,1].plot(df.index, df['system_mean_waiting_time'], alpha=0.5, label='Raw Data')
    axes[1,1].plot(df.index, moving_avg, linewidth=3, label=f'Moving Average (window={window_size})')
    axes[1,1].set_title('Mean Waiting Time with Trend', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Mean Waiting Time (s)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    # create output directory
    SAVE_DIR = os.path.join(save_dir, algo_name, "waiting_time", "combined")
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(f"{SAVE_DIR}/combined_time_series.png", dpi=300, bbox_inches='tight')
    plt.show()

def _argparse() -> argparse.Namespace:
    """
        Parse given args from the terminal and return them to the program.
    """
    parser = argparse.ArgumentParser(description="Plot waiting time metrics from the csv file for the given algorithm")
    parser.add_argument("algo", help="Algorithm used for evaluation (ppo/dqn/sac)")
    return parser.parse_args()

def main():
    # argparse
    args = _argparse()
    algo = args.algo
    # Load and process data
    log_file = f"logs_conn1_ep991_iter499.csv"
    df = pd.read_csv(os.path.join(LOGS_DIR, algo, log_file))

    # Filter needed columns only
    df = df[NEEDED_COL_NAMES]

    # Create enhanced plots
    plot_time_series_individual(df, algo)
    plot_time_series_combined(df, algo)

if __name__ == "__main__":
    main()