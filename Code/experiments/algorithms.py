# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.tune.registry import register_env

# import sumo_rl # type: ignore


# class AlgoConfigFactory:
#     """
#         Common class to define ray rl algorithms. 
#     """
#     def __init__(self, env_config: dict): # no need to pass algo specific args here. add them in each separate method below.
#         self.env_config = env_config

#     # create env (helper)
#     def _create_env(self, config = None):
#         """
#             Create a parallel env in SUMO.
#             config = None was added to make this compatible with the lambda in registry
#         """
#         return sumo_rl.parallel_env(**self.env_config if config is None else config)
    
#     # All algo. configs.

#     # ! 1) PPO
#     def get_ppo_config(self, ppo_hparams: dict):

#         return (
            
#             PPOConfig() # ! This structure is called method chaining fyi 
#             .environment(
#                 env = "sumo_multi_agent",
#                 env_config = self.env_config
#             )
#             .framework('torch')
#             .env_runners(
#                 num_env_runners = 1, # num of parallel runners
#                 # rollout_fragment_length = 200 # episodes are broken down into chunks of this size
#             )
#             .learners(
#                 num_learners=1,
#             )
#             .training(
#                 **ppo_hparams, # unpack the training hyperparams
#             )
#             # .evaluation(

#             # )
#             .multi_agent(
#                 policies={"shared_policy": (None, None, None, {})},
#                 policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
#             )
#             # .rl_module(
#             #     rl_module_spec=MultiRLModuleSpec(
#             #         # All agents (0 and 1) use the same (single) RLModule.
#             #         # rl_module_specs=RLModuleSpec(
#             #         #     module_class=MyRLModuleClass,
#             #         #     model_config={"some_key": "some_setting"},
#             #         # )
#             #     )
#             # )   
#         )

#     # ! 2) DQN 
#     def get_dqn_config(self, dqn_hparams: dict):


#         pass

#     # ..

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.typing import PolicyID
# from ray.tune.registry import register_env
# from typing import Dict, Tuple
# import gymnasium as gym

import sumo_rl


class AlgoConfigFactory:
    """
    Common class to define ray rl algorithms with CTDE support.
    """
    def __init__(self, env_config: dict):
        self.env_config = env_config

    # create env (helper)
    def _create_env(self, config = None):
        """
            Create a parallel env in SUMO.
            config = None was added to make this compatible with the lambda in registry
        """
        return sumo_rl.parallel_env(**self.env_config if config is None else config)
    
    def _get_obs_and_action_spaces(self):
        """Get observation and action spaces from environment"""
        temp_env = self._create_env()
        temp_env.reset()
        
        # Get spaces from any agent (assuming homogeneous agents)
        agent_ids = list(temp_env.agents)
        sample_agent = agent_ids[0]
        
        obs_space = temp_env.observation_space(sample_agent)
        action_space = temp_env.action_space(sample_agent)
        
        temp_env.close()
        return obs_space, action_space, agent_ids

    def get_ppo_config(self, ppo_hparams: dict):
        """
        Create PPO configuration with CTDE support
        """
        obs_space, action_space, agent_ids = self._get_obs_and_action_spaces()
    
        config = (
            PPOConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1)
            .learners(num_learners=1)
            .training(**ppo_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )

        # CTDE: Single shared policy for all agents (parameter sharing)
        # Training aggregates rollouts from all agents (centralized).
        # Execution: Each agent inputs its local observation to the shared policy (decentralized).
        shared_policy_spec = PolicySpec(
            observation_space=obs_space,
            action_space=action_space,
            config={}  
        )
        
        config = config.multi_agent(
            policies={"shared_policy": shared_policy_spec},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )

        return config

    # def get_dqn_config(self, dqn_hparams: dict):
    #     """
    #     Create DQN configuration for multi-agent learning
    #     """
    #     obs_space, action_space, agent_ids = self._get_obs_and_action_spaces()
        
    #     return (
    #         DQNConfig()
    #         .environment(
    #             env="sumo_multi_agent",
    #             env_config=self.env_config
    #         )
    #         .framework('torch')
    #         .env_runners(
    #             num_env_runners=2,
    #             rollout_fragment_length=4  # Smaller for DQN
    #         )
    #         .learners(
    #             num_learners=1,
    #         )
    #         .training(
    #             **dqn_hparams,
    #         )
    #         .evaluation(
    #             evaluation_interval=10,
    #             evaluation_duration=10,
    #         )
    #         .multi_agent(
    #             policies={"shared_policy": PolicySpec(
    #                 observation_space=obs_space,
    #                 action_space=action_space,
    #             )},
    #             policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    #         )
    #     )

    def get_policy_mapping_fn(self, agent_ids: list, use_shared_policy: bool = True):
        """Helper function to create policy mapping"""
        if use_shared_policy:
            return lambda agent_id, *args, **kwargs: "shared_policy"
        else:
            return lambda agent_id, *args, **kwargs: f"policy_{agent_id}"


