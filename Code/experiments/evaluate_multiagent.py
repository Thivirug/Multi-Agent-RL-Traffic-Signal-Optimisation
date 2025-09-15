# experiments/evaluate_checkpoint.py
import os
import ray
from Code.config import ENV_CONFIG
from algorithms import AlgoConfigFactory
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.algorithm import Algorithm

def main(checkpoint_dir: str):
    # Initialize Ray
    ray.init()

    # Create factory and register environment
    # update env config to use gui and increase sim duration
    ENV_CONFIG.update([('use_gui', True), ('num_seconds', 5000)])
    factory = AlgoConfigFactory(ENV_CONFIG)
    register_env(
        name="sumo_multi_agent",
        env_creator=lambda config: ParallelPettingZooEnv(factory._create_env(config))
    )

    # Restore the algorithm from checkpoint
    # checkpoint_dir: Full path, e.g., "Code/outputs/checkpoints/ppo/100"
    algo = Algorithm.from_checkpoint(checkpoint_dir)  

    # Run evaluation
    results = algo.evaluate()

    # Print key metrics
    print(f"Evaluation Results for {checkpoint_dir}:")
    print(f"  Mean Episode Reward: {results['env_runners']['episode_return_mean']:.5f}")
    print(f"  Per-Agent Mean Rewards: {results['env_runners'].get('agent_episode_returns_mean', {})}")
    print(f"  Episode Length Mean: {results['env_runners']['episode_len_mean']:.2f}")

    # Save results to JSON
    # import json
    # with open(os.path.join(os.path.dirname(checkpoint_dir), f"eval_results_{os.path.basename(checkpoint_dir)}.json"), "w") as f:
    #     json.dump(results, f, indent=2)
    # print(f"Results saved to eval_results_{os.path.basename(checkpoint_dir)}.json")

    # Stop the algorithm
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    # Example usage: Replace with your checkpoint path
    checkpoint_path = os.path.abspath("Code/outputs/checkpoints/ppo/100")  # final checkpoint
    main(checkpoint_path)  # Evaluate 20 episodes