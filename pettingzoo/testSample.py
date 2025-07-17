import pettingzoo
from pettingzoo.classic import tictactoe_v3
import random
import numpy as np

def mainSimple():
    # 创建井字棋环境
    env = tictactoe_v3.env(render_mode="human")
    # 重置环境以开始新的游戏
    env.reset(seed=42)
    # 循环直到游戏结束
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # 随机选择一个可用的动作
            action = np.random.choice(np.where(mask)[0])

        env.step(action)

    # 渲染最终的环境状态
    env.render()

def simple_policy(observation):
    action_mask = observation['action_mask']
    # 尝试在中心位置放置标记
    if action_mask[4]:
        return 4
    # 如果中心位置已被占用，尝试在角落位置放置标记
    corners = [0, 2, 6, 8]  # 角落位置对应的action编号
    for corner in corners:
        if action_mask[corner]:
            return corner
    # 如果所有首选位置都被占用，选择一个可用位置
    available_actions = np.where(action_mask)[0]
    return available_actions[0]  # 返回第一个可用的动作

def mainPolicy():
    # 创建井字棋环境
    env = tictactoe_v3.env(render_mode="human")
    # 重置环境以开始新的游戏
    env.reset(seed=42)
    # 循环直到游戏结束
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = simple_policy(observation)  # 调用simple_policy函数来确定动作

        env.step(action)

    # 渲染最终的环境状态
    env.render()

if __name__ == "__main__":
    # mainSimple()
    mainPolicy()
