import sumo_rl # type: ignore
 
env = sumo_rl.parallel_env(net_file='sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='sumo-rl/sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)