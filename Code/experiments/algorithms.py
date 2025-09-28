import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# config imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig

from ray.rllib.policy.policy import PolicySpec
import sumo_rl # type: ignore

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
import pettingzoo
from gymnasium.spaces import Space

# ! NOTE :         
# Using CTDE: Single shared policy for all agents (parameter sharing)
# Training aggregates rollouts from all agents (centralized).
# Execution: Each agent inputs its local observation to the shared policy (decentralized). - Done in evaluate_multiagent.py

class AlgoConfigFactory:
    """
        Common class to define ray rl algorithms with CTDE support.
    """
    def __init__(self, env_config: dict) -> None:
        self.env_config = env_config

    # create env (helper)
    def _create_env(self, config: dict = None) -> pettingzoo.utils.conversions.aec_to_parallel_wrapper:
        """
            Create a parallel env in SUMO.

            Args:
                config (dict, optional): Environment configuration dictionary. Defaults to None.
                    If None, uses self.env_config.
            > NOTE:  config = None was added to make this compatible with the lambda in registry

            Returns:
                A parallel environment instance from sumo_rl.
        """
        return sumo_rl.parallel_env(**self.env_config if config is None else config)
    
    def _get_obs_and_action_spaces(self) -> tuple[Space, Space]:
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
        return obs_space, action_space

    def _get_shared_policy_spec(self) -> PolicySpec:
        """
            Return the PolicySpec object to be passed into the shared policy map.
        """
        obs_space, action_space = self._get_obs_and_action_spaces()

        # policy spec definition
        shared_policy_spec = PolicySpec(
            observation_space=obs_space,
            action_space=action_space,
            config={}  
        )

        return shared_policy_spec
    
    def _CTDE_config(self, config: AlgorithmConfig) -> AlgorithmConfig:
        """
            Return the updated config with CTDE support. 

            Args: 
                config (AlgorithmConfig): The base algorithm config to be updated.
            
            Returns:
                The updated config with CTDE support.

            Raises:
                Exception: If the policy spec or environment does not support multi-agent.
        """
        try:
            return config.multi_agent(
                policies={"shared_policy": self._get_shared_policy_spec()},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
        except:
            raise Exception("Check policy spec or env support for MARL")

    # ! ================== PPO ==================
    def get_ppo_config(self, ppo_hparams: dict) -> AlgorithmConfig:
        """
            Create PPO configuration with CTDE support

            Args:
                ppo_hparams (dict): Hyperparameters for PPO training.

            Returns:
                The PPO configuration with CTDE support.
        """

        # create config
        config = (
            PPOConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, num_gpus_per_env_runner=1, rollout_fragment_length="auto")
            .learners(num_learners=2, num_cpus_per_learner=5)
            .training(**ppo_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )
        
        # creating CTDE shared policy 
        return self._CTDE_config(config)

    # ! ================== DQN ==================
    def get_dqn_config(self, dqn_hparams: dict) -> AlgorithmConfig:
        """
            Create DQN configuration with CTDE support.

            Args:
                dqn_hparams (dict): Hyperparameters for DQN training.

            Returns:
                The DQN configuration with CTDE support.
        """

        # create config
        config = (
            DQNConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, rollout_fragment_length="auto") # ! CHANGE THESE FOR UR PC REQS
            .learners(num_learners=1, num_cpus_per_learner=2) # ! CHANGE THESE FOR UR PC REQS
            .training(**dqn_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )
        
        # creating CTDE shared policy 
        return self._CTDE_config(config)
    
    # ! ================== SAC ==================
    def get_sac_config(self, sac_hparams: dict) -> AlgorithmConfig:
        """
            Create SAC configuration with CTDE support.

            Args:
                sac_hparams (dict): Hyperparameters for SAC training.

            Returns:
                The SAC configuration with CTDE support.
        """

        # create config
        config = (
            SACConfig()
            .environment(env="sumo_multi_agent", env_config=self.env_config)
            .framework('torch')
            .env_runners(num_env_runners=1, num_gpus_per_env_runner=1, rollout_fragment_length="auto") # ! CHANGE THESE FOR UR PC REQS
            .learners(num_learners=1, num_cpus_per_learner=4, num_gpus_per_learner=1) # ! CHANGE THESE FOR UR PC REQS
            .training(**sac_hparams)
            .evaluation(
                evaluation_interval=10,
                evaluation_duration=10,
                evaluation_config={"env_config": self.env_config}
            )
        )
        
        # creating CTDE shared policy 
        return self._CTDE_config(config)
    

            
