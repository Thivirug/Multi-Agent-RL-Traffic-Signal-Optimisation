from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl # type: ignore


class AlgoConfigFactory:
    """
        Common class to define ray rl algorithms. 
    """
    def __init__(self, env_config: dict): # no need to pass algo specific args here. add them in each separate method below.
        self.env_config = env_config

    # create env (helper)
    def _create_env(self, config = None):
        """
            Create a parallel env in SUMO.
            config = None was added to make this compatible with the lambda in registry
        """
        return sumo_rl.parallel_env(**self.env_config if config is None else config)
    
    # All algo. configs.

    # ! 1) PPO
    def get_ppo_config(self, ppo_hparams: dict):

        return (
            
            PPOConfig() # ! This structure is called method chaining fyi 
            .environment(
                env = "sumo_multi_agent",
                env_config = self.env_config
            )
            .framework('torch')
            .env_runners(
                num_env_runners = 1, # num of parallel runners
                # rollout_fragment_length = 200 # episodes are broken down into chunks of this size
            )
            .learners(
                num_learners=1,
            )
            .training(
                **ppo_hparams, # unpack the training hyperparams
            )
            # .evaluation(

            # )
            .multi_agent(
                policies={"shared_policy": (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
            # .rl_module(
            #     rl_module_spec=MultiRLModuleSpec(
            #         # All agents (0 and 1) use the same (single) RLModule.
            #         # rl_module_specs=RLModuleSpec(
            #         #     module_class=MyRLModuleClass,
            #         #     model_config={"some_key": "some_setting"},
            #         # )
            #     )
            # )   
        )

    # ! 2) DQN 
    def get_dqn_config(self, dqn_hparams: dict):


        pass

    # ..


