import numpy as np

from pettingzoo.mpe import simple_tag_v3

def originAction():
    env = simple_tag_v3.env(render_mode="human")
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

def optAction():
    env = simple_tag_v3.env(render_mode="human", max_cycles=500)
    env.reset(seed=42)

    def get_distance(pos1, pos2):
        """计算两个位置之间的欧几里得距离"""
        return np.sqrt(sum((pos1 - pos2) ** 2))

    def predator_policy(observation, agent):
        """捕食者策略：追逐最近的猎物"""
        # 观测空间结构：[自身位置(2), 自身速度(2), 其他智能体位置(6), 地标位置(4), 猎物位置(2)]
        prey_pos = observation[-2:]  # 猎物的位置
        predator_pos = observation[:2]  # 捕食者自身位置
        
        # 计算朝向猎物的方向向量
        direction = prey_pos - predator_pos
        distance = get_distance(prey_pos, predator_pos)
        
        # 标准化方向向量
        if distance > 0:
            direction = direction / distance
        
        # 动作空间：[无动作, 左, 右, 上, 下]
        # 我们将方向向量转换为离散动作
        if abs(direction[0]) > abs(direction[1]):
            action = 2 if direction[0] > 0 else 1  # 右或左
        else:
            action = 3 if direction[1] > 0 else 4  # 上或下
        
        # 如果已经很接近猎物，可以减速
        if distance < 0.3:
            action = 0  # 无动作
        
        return action

    def prey_policy(observation, agent):
        """猎物策略：逃离最近的捕食者并躲避障碍物"""
        # 观测空间结构：[自身位置(2), 自身速度(2), 其他智能体位置(6), 地标位置(4)]
        predator_positions = observation[4:10].reshape(3, 2)  # 3个捕食者的位置
        prey_pos = observation[:2]  # 猎物自身位置
        landmark_pos = observation[10:14].reshape(2, 2)  # 2个地标的位置
        
        # 找到最近的捕食者
        distances = [get_distance(prey_pos, p_pos) for p_pos in predator_positions]
        closest_predator_idx = np.argmin(distances)
        closest_predator_pos = predator_positions[closest_predator_idx]
        
        # 计算逃离方向（远离最近的捕食者）
        escape_direction = prey_pos - closest_predator_pos
        
        # 考虑躲避地标
        for landmark in landmark_pos:
            landmark_dist = get_distance(prey_pos, landmark)
            if landmark_dist < 0.5:  # 如果太靠近地标
                # 添加远离地标的方向
                escape_direction += (prey_pos - landmark) * (0.5 / landmark_dist)
        
        # 标准化方向向量
        norm = np.linalg.norm(escape_direction)
        if norm > 0:
            escape_direction = escape_direction / norm
        
        # 将方向向量转换为离散动作
        if abs(escape_direction[0]) > abs(escape_direction[1]):
            action = 2 if escape_direction[0] > 0 else 1  # 右或左
        else:
            action = 3 if escape_direction[1] > 0 else 4  # 上或下
        
        # 随机性增加逃脱机会
        if np.random.random() < 0.1:
            action = np.random.randint(1, 5)
        
        return action

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
    env.close()

if __name__ == '__main__':
    optAction()
