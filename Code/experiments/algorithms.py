from code.config import ENV_CONFIG

import ray
from ray.rllib.algorithms.ppo import PPOConfig


# 1) PPO
print(ENV_CONFIG)