import numpy as np
import time

from pettingzoo.utils.conversions import parallel_wrapper_fn

import ModifyEnv

env = ModifyEnv.env(render_mode = "human")
parallel_env = parallel_wrapper_fn(env)

num_landmarks = 0
num_good = 0
num_adversaries = 0
num_agents = 0

def originAction():
    env.reset(seed = 42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()

def originActionParallel():
    observations, infos = parallel_env.reset(seed = 42)

    while parallel_env.agents:
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    
    parallel_env.close()

##################################
def get_distance(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return np.sqrt(sum((pos1 - pos2) ** 2))

def predator_policy(observation, agent):
    """
    捕食者策略：协作追逐猎物
    观测空间结构：[自身速度(2), 自身位置(2), 地标位置(2*2), 其他智能体位置(3*2), 猎物速度(2)]
    """
    
    # 解析观测空间
    agent_vel = observation[:2]
    agent_pos = observation[2:4]
    landmark_pos = observation[4: 8].reshape(num_landmarks, 2)  # 地标
    other_pos = observation[8 : 14].reshape(num_agents - 1, 2)   # 其他智能体位置
    prey_vel = observation[14 :]                # 猎物的速度
    
    # 找到猎物的位置
    prey_pos = other_pos[-1]
    # if prey_pos is None:
    #     return env.action_space(agent).sample()  # 如果没有找到猎物，随机移动
    
    # 计算朝向猎物的方向向量（考虑预测猎物移动）
    predicted_prey_pos = prey_pos + prey_vel * 0.5  # 简单的线性预测
    direction = predicted_prey_pos - agent_pos
    distance = get_distance(predicted_prey_pos, agent_pos)
    
    # 标准化方向向量
    if distance > 0:
        direction = direction / distance
    
    # 考虑与其他捕食者的协作（避免全部聚集在同一个位置）
    for other in other_pos:
        if np.allclose(other, prey_pos):
            continue  # 跳过猎物
        other_dist = get_distance(agent_pos, other)
        if other_dist < 1.0:  # 如果太靠近其他捕食者
            # 添加一点分离力
            direction += (agent_pos - other) * 0.3
    
    # 考虑躲避地标
    for landmark in landmark_pos:
        landmark_dist = get_distance(agent_pos, landmark)
        if landmark_dist < 0.5:  # 如果太靠近地标
            direction += (agent_pos - landmark) * (0.5 / landmark_dist)
    
    # 考虑躲避边界 
    next_pos = agent_pos + direction
    if next_pos[0] <= -1 or next_pos[0] >= 1:
        direction[0] = -direction[0]
    if next_pos[1] <= -1 or next_pos[1] >= 1:
        direction[1] = -direction[1]

    # 标准化最终方向向量
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    # 将方向向量转换为离散动作
    if abs(direction[0]) > abs(direction[1]):
        action = 2 if direction[0] > 0 else 1  # 右或左
    else:
        # action = 3 if direction[1] > 0 else 4  # 上或下
        action = 4 if direction[1] > 0 else 3
    # 如果已经很接近猎物，可以减速
    if distance < 0.3:
        action = 0  # 无动作
    
    return action

def prey_policy(observation, agent):
    """
    猎物策略：逃离捕食者并躲避障碍物
    观测空间结构：[自身速度(2), 自身位置(2), 地标位置(2*2), 其他智能体位置(3*2)]
    """
    # 解析观测空间
    agent_vel = observation[:2]
    agent_pos = observation[2:4]
    landmark_pos = observation[4:8].reshape(2, 2)  # 2个地标
    other_pos = observation[8:14].reshape(3, 2)    # 3个其他智能体位置
    
    # 找到所有捕食者的位置（捕食者是对抗智能体）
    # 按顺序，先adversary后agent
    predator_positions = other_pos[:]
    # for pos in other_pos:
    #     if any(np.allclose(pos, p) for p in other_pos if not np.array_equal(pos, p)):
    #         predator_positions.append(pos)
    
    # if not predator_positions:
    #     return env.action_space(agent).sample()  # 如果没有捕食者，随机移动
    
    # 计算逃离方向（远离所有捕食者的综合方向）
    escape_direction = np.zeros(2)
    for pred_pos in predator_positions:
        pred_dist = get_distance(agent_pos, pred_pos)
        escape_direction += (agent_pos - pred_pos) / max(pred_dist, 0.1)  # 距离越近权重越大
    
    # 考虑躲避地标
    for landmark in landmark_pos:
        landmark_dist = get_distance(agent_pos, landmark)
        if landmark_dist < 0.5:  # 如果太靠近地标
            escape_direction += (agent_pos - landmark) * (0.5 / landmark_dist)

    # 添加一些随机性
    # escape_direction += np.random.uniform(-0.3, 0.3, size=2)
    
    # 标准化方向向量
    norm = np.linalg.norm(escape_direction)
    if norm > 0:
        escape_direction = escape_direction / norm

    # 考虑躲避边界
    next_pos = agent_pos + escape_direction
    if next_pos[0] <= -1 or next_pos[0] >= 1:
        escape_direction[0] = -escape_direction[0]
    if next_pos[1] <= -1 or next_pos[1] >= 1:
        escape_direction[1] = -escape_direction[1]
    # print(agent_pos, escape_direction, next_pos, agent_pos + escape_direction)
    
    # 将方向向量转换为离散动作
    if abs(escape_direction[0]) > abs(escape_direction[1]):
        action = 2 if escape_direction[0] > 0 else 1  # 右或左
    else:
        # action = 3 if escape_direction[1] > 0 else 4  # 上或下
        action = 4 if escape_direction[1] > 0 else 3
    # 如果有捕食者很近，增加移动概率
    # min_pred_dist = min(get_distance(agent_pos, p) for p in predator_positions)
    # if min_pred_dist < 0.5 and action == 0:
    #     action = np.random.randint(1, 5)
    return action

def optAction():
    # num_good=1, num_adversaries=3, num_obstacles=2
    env.reset(seed = 40)

    global num_landmarks
    global num_adversaries
    global num_agents
    global num_good

    num_landmarks = len(env.world.landmarks)
    num_agents = len(env.world.agents)
    num_good = sum([1 for agent in env.agents if 'agent' in agent])
    num_adversaries = num_agents - num_good

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            if "adversary" in agent:  # 捕食者
                action = predator_policy(observation, agent)
            else:  # 猎物
                action = prey_policy(observation, agent)
        
        env.step(action)
    # time.sleep(3)
    env.close()

def optActionParallel():
    from pettingzoo.mpe import simple_tag_v3
    parallel_env = simple_tag_v3.parallel_env(render_mode="human")
    observations, infos = parallel_env.reset(seed = 42)
    
    while parallel_env.agents:
        # actions = {agent: prey_policy(observations, agent) for agent in parallel_env.agents if 'agent' in agent else predator_policy(observations, agent)}
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
        # observations: <class 'dict'> {'adversary_0': array...
        # {'adversary_0': False, 'adversary_1': False, 'adversary_2': False, 'agent_0': False} {'adversary_0': False, 'adversary_1': False, 'adversary_2': False, 'agent_0': False}
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    parallel_env.close()


if __name__ == '__main__':
    optAction()
    # originAction()
    # optActionParallel()