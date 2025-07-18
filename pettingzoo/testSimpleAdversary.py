# from pettingzoo.sisl import pursuit_v4
from pettingzoo.mpe import simple_adversary_v3

# env = pursuit_v4.env(render_mode="human")
env = simple_adversary_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()