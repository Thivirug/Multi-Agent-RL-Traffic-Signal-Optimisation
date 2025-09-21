# to observe env info like action space, obs space, agents etc
import sumo_rl # type: ignore

env = sumo_rl.parallel_env(net_file='src/sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml',
                  route_file='src/sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml',
                  use_gui=False,
                  num_seconds=3000)

# Reset the environment to start a new simulation
observations, _ = env.reset()

print("Number of agents:", len(env.agents))
print("Agents:", env.agents)
print("Action Space:", env.action_space(env.agents[0]))
print("Observation Space:", env.observation_space(env.agents[0]))
print("Observation for one agent:", observations[env.agents[0]])
print("Shape of Observation for one agent:", observations[env.agents[0]].shape)
print("Dtype of Observation for one agent:", observations[env.agents[0]].dtype)
print("Low of Observation for one agent:", env.observation_space(env.agents[0]).low)
print("High of Observation for one agent:", env.observation_space(env.agents[0]).high)
print("Sample Action for one agent:", env.action_space(env.agents[0]).sample())
print("Type of Action Space for one agent:", type(env.action_space(env.agents[0])))
print("Type of Observation Space for one agent:", type(env.observation_space(env.agents[0])))
print("Is Action Space Discrete for one agent:", env.action_space(env.agents[0]).__class__.__name__ == "Discrete")
print("Is Observation Space Box for one agent:", env.observation_space(env.agents[0]).__class__.__name__ == "Box")
print("Max of Action Space for one agent:", env.action_space(env.agents[0]).n - 1)  # For Discrete space, max is n-1
print("Min of Action Space for one agent:", 0)  # For Discrete space
