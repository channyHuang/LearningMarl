from pettingzoo.mpe import simple_adversary_v3
# from pettingzoo.mpe import simple_tag_v3

parallel_env = simple_adversary_v3.parallel_env(render_mode="human")
# parallel_env = simple_tag_v3.parallel_env(render_mode="human")
observations, infos = parallel_env.reset(seed=42)

while parallel_env.agents:
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)

parallel_env.close()
