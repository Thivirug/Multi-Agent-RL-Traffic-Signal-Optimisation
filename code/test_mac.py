import sumo_rl
from importlib import resources  # Python 3.9+

# Use the files that come with the installed sumo_rl package
net = resources.files(sumo_rl) / "nets/RESCO/grid4x4/grid4x4.net.xml"
rou = resources.files(sumo_rl) / "nets/RESCO/grid4x4/grid4x4_1.rou.xml"

env = sumo_rl.parallel_env(
    net_file=str(net),
    route_file=str(rou),
    use_gui=True,   # set False for a quick sanity run; switch to True after it works
    num_seconds=3600
)

obs = env.reset()
print("Num of agents:", len(env.agents))

while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, term, trunc, info = env.step(actions)
