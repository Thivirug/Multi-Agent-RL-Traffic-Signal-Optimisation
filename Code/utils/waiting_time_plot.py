import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# print(f"There are {len(os.listdir(LOGS_DIR))} log files")

# # pattern = re.compile(r"(logs_conn\d+_ep\d+)\.csv")

# for log in os.listdir(LOGS_DIR):
#     pass

# plotting for a given single log file
df = pd.read_csv(os.path.join(LOGS_DIR, "logs_conn1_ep391_iter199.csv"))
print(df)

# filter needed columns only
df = df[NEEDED_COL_NAMES]
print(df)

# separate plots
for col in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y=col)
    plt.title(f"Waiting Time Metrics Over Time - {col}")
    plt.xlabel("Time Step")
    plt.ylabel("Waiting Time (seconds)")
    plt.tight_layout()
    plt.show()