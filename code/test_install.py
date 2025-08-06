import sumo_rl # type: ignore

# This script is used to test the installation of the sumo_rl package.
# It creates a parallel environment with a grid network and runs a simple simulation.

env = sumo_rl.parallel_env(net_file='sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml',
                  route_file='sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml',
                  use_gui=True,
                  num_seconds=2000)

# Reset the environment to start a new simulation
observations = env.reset()

print("Nu of agents:", len(env.agents))  

while env.agents: # Continue until all agents are done
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # Sample random actions for each agent
    observations, rewards, terminations, truncations, infos = env.step(actions) # Step the environment with the sampled actions