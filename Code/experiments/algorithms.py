import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# config imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig

from ray.rllib.policy.policy import PolicySpec
import sumo_rl # type: ignore

# ! NOTE :         
# Using CTDE: Single shared policy for all agents (parameter sharing)
# Training aggregates rollouts from all agents (centralized).
# Execution: Each agent inputs its local observation to the shared policy (decentralized). - Done in evaluate_multiagent.py

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
        """
            Get observation and action spaces from environment.
        """

        # create a temp env obj
        temp_env = self._create_env()
        temp_env.reset()
        
        # Get spaces from any agent 
        agent_ids = list(temp_env.agents)
        sample_agent = agent_ids[0]
        obs_space = temp_env.observation_space(sample_agent)
        action_space = temp_env.action_space(sample_agent)
        
        temp_env.close() 
        return obs_space, action_space, agent_ids

    # ! ================== PPO ==================
    def get_ppo_config(self, ppo_hparams: dict):
        """
            Create PPO configuration with CTDE support
        """
        obs_space, action_space, _ = self._get_obs_and_action_spaces()

        # create config
        config = (
            PPOConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, num_gpus_per_env_runner=1)
            .learners(num_learners=2, num_cpus_per_learner=7)
            .training(**ppo_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )

        # policy spec definition
        shared_policy_spec = PolicySpec(
            observation_space=obs_space,
            action_space=action_space,
            config={}  
        )
        
        # creating shared policy
        config = config.multi_agent(
            policies={"shared_policy": shared_policy_spec},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )

        return config

    # ! ================== DQN ==================
    def get_dqn_config(self, dqn_hparams: dict):
        """
            Create DQN configuration with CTDE support.
        """
        obs_space, action_space, agent_ids = self._get_obs_and_action_spaces()

        obs_space, action_space, _ = self._get_obs_and_action_spaces()

        # create config
        config = (
            DQNConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, num_gpus_per_env_runner=1) # ! CHANGE THESE FOR UR PC REQS
            .learners(num_learners=2, num_cpus_per_learner=7) # ! CHANGE THESE FOR UR PC REQS
            .training(**dqn_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )

        # policy spec definition
        shared_policy_spec = PolicySpec(
            observation_space=obs_space,
            action_space=action_space,
            config={}  
        )
        
        # creating shared policy
        config = config.multi_agent(
            policies={"shared_policy": shared_policy_spec},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )

        return config
    
    # ! ================== SAC ==================
    def get_sac_config(self, sac_hparams: dict):
        """
            Create SAC configuration with CTDE support.
        """
        obs_space, action_space, agent_ids = self._get_obs_and_action_spaces()

        obs_space, action_space, _ = self._get_obs_and_action_spaces()

        # create config
        config = (
            SACConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, num_gpus_per_env_runner=1) # ! CHANGE THESE FOR UR PC REQS
            .learners(num_learners=2, num_cpus_per_learner=7) # ! CHANGE THESE FOR UR PC REQS
            .training(**sac_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )

        # policy spec definition
        shared_policy_spec = PolicySpec(
            observation_space=obs_space,
            action_space=action_space,
            config={}  
        )
        
        # creating shared policy
        config = config.multi_agent(
            policies={"shared_policy": shared_policy_spec},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )

        return config
    


