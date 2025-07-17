'''
自定义environment，基于MADDPG写一个多智能体巡逻算法,要求如下：
1. 自定义地图，障碍物占比约30%
2. 有n个智能体，n>=3
3. 智能体在一定时间内合作完成巡逻任务，每个智能体巡逻半径为r
4. action为连续空间，policy通过训练生成模型
5. 能够有一个函数验证训练好的模型，验证时使用pygame实时render可视化巡逻过程
'''


import pygame
import numpy as np
import random
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class MultiAgentPatrolEnv:
    def __init__(self, map_size=(15, 15), num_agents=4, max_steps=500):
        """
        初始化多智能体巡逻环境
        
        参数:
        map_size: 地图尺寸 (width, height)
        num_agents: 智能体数量 (n >= 3)
        max_steps: 单次训练最大步数
        """
        self.width, self.height = map_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agent_positions = []
        self.visited = defaultdict(int)  # 记录位置访问次数
        self.steps = 0
        self.total_visits = 0
        self.unique_visits = set()
        
        # 创建自定义地图 (0=可通行区域, 1=障碍物)
        self.map = self.generate_map()
        
        # 动作空间: [上, 右, 下, 左, 停留]
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        
        # 状态空间大小 (智能体位置 + 局部地图视图)
        self.state_size = 2 + 9  # 自身坐标 + 3x3局部地图
        
        # 初始化智能体位置
        self.reset()
        
        # PyGame 可视化设置
        self.cell_size = 30
        self.screen_width = self.width * self.cell_size
        self.screen_height = self.height * self.cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("多智能体巡逻模拟")
        
        # 颜色定义
        self.COLORS = {
            'background': (240, 240, 240),
            'obstacle': (50, 50, 50),
            'agent': [(220, 60, 60), (60, 150, 220), (220, 180, 60), 
                      (160, 90, 180), (70, 180, 110), (220, 120, 170),
                      (80, 200, 200), (200, 100, 50), (150, 150, 150)],
            'visited': (180, 220, 240),
            'text': (30, 30, 30),
            'path': (200, 200, 200)
        }

    def generate_map(self, obstacle_ratio=0.15):
        """生成自定义地图，包含随机障碍物"""
        map_grid = np.zeros((self.height, self.width), dtype=int)
        
        # 添加随机障碍物
        for _ in range(int(self.width * self.height * obstacle_ratio)):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            map_grid[y][x] = 1
        
        # 确保至少有一条路径可以访问所有区域
        map_grid[0, :] = 0  # 顶部行保持畅通
        map_grid[:, 0] = 0  # 左侧列保持畅通
        map_grid[-1, :] = 0  # 底部行保持畅通
        map_grid[:, -1] = 0  # 右侧列保持畅通
        
        # 添加一些房间结构
        for _ in range(3):
            room_w = random.randint(3, self.width//3)
            room_h = random.randint(3, self.height//3)
            room_x = random.randint(1, self.width - room_w - 1)
            room_y = random.randint(1, self.height - room_h - 1)
            
            # 创建房间
            map_grid[room_y:room_y+room_h, room_x:room_x+room_w] = 0
            
            # 在房间墙上开一个门
            wall_side = random.choice(['top', 'right', 'bottom', 'left'])
            if wall_side == 'top' and room_y > 0:
                door_x = room_x + random.randint(1, room_w-2)
                map_grid[room_y, door_x] = 0
            elif wall_side == 'right' and room_x+room_w < self.width-1:
                door_y = room_y + random.randint(1, room_h-2)
                map_grid[door_y, room_x+room_w-1] = 0
            elif wall_side == 'bottom' and room_y+room_h < self.height-1:
                door_x = room_x + random.randint(1, room_w-2)
                map_grid[room_y+room_h-1, door_x] = 0
            elif wall_side == 'left' and room_x > 0:
                door_y = room_y + random.randint(1, room_h-2)
                map_grid[door_y, room_x] = 0
        
        return map_grid

    def reset(self):
        """重置环境"""
        self.agent_positions = []
        self.visited.clear()
        self.unique_visits.clear()
        self.total_visits = 0
        self.steps = 0
        
        # 在可通行区域随机放置智能体
        while len(self.agent_positions) < self.num_agents:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.map[y][x] == 0 and (x, y) not in self.agent_positions:
                self.agent_positions.append((x, y))
                self.record_visit(x, y)
        
        # 返回初始状态
        return self.get_state()
    
    def record_visit(self, x, y):
        """记录位置访问情况"""
        self.visited[(x, y)] += 1
        self.unique_visits.add((x, y))
        self.total_visits += 1

    def get_state(self):
        """获取所有智能体的状态"""
        states = []
        for i, (x, y) in enumerate(self.agent_positions):
            # 自身位置
            agent_state = [x / self.width, y / self.height]
            
            # 3x3局部地图视图
            local_view = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        # 0: 可通行, 1: 障碍物, 0.5: 边界外视为障碍物
                        local_view.append(self.map[ny][nx])
                    else:
                        local_view.append(1)  # 边界外视为障碍物
            states.append(agent_state + local_view)
        
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        """执行智能体动作"""
        self.steps += 1
        rewards = [0] * self.num_agents
        collisions = 0
        
        # 更新每个智能体的位置
        new_positions = []
        for i, (x, y) in enumerate(self.agent_positions):
            action = actions[i]
            
            # 计算新位置
            new_x, new_y = x, y
            if action == 0 and y > 0:  # 上
                new_y = y - 1
            elif action == 1 and x < self.width - 1:  # 右
                new_x = x + 1
            elif action == 2 and y < self.height - 1:  # 下
                new_y = y + 1
            elif action == 3 and x > 0:  # 左
                new_x = x - 1
            
            # 检查移动是否有效（不撞墙）
            if 0 <= new_x < self.width and 0 <= new_y < self.height and self.map[new_y][new_x] == 0:
                x, y = new_x, new_y
            new_positions.append((x, y))
            self.record_visit(x, y)
            
            # 奖励计算
            visit_count = self.visited[(x, y)]
            if visit_count == 1:  # 首次访问
                rewards[i] = 1.0
            elif visit_count == 2:  # 第二次访问
                rewards[i] = -0.2
            else:  # 多次访问
                rewards[i] = -0.5
            
            # 碰撞惩罚
            if new_positions.count((x, y)) > 1:
                rewards[i] -= 0.5
                collisions += 1
        
        self.agent_positions = new_positions
        
        # 检查终止条件
        done = self.steps >= self.max_steps
        
        # 覆盖率指标
        total_cells = self.width * self.height - np.sum(self.map)
        coverage = len(self.unique_visits) / total_cells if total_cells > 0 else 0.0
        
        # 额外奖励：当覆盖率达到一定阈值
        if coverage > 0.8:
            for i in range(self.num_agents):
                rewards[i] += 0.1
        
        # 团队奖励：鼓励探索
        team_reward = coverage * 0.5 - collisions * 0.1
        rewards = [r + team_reward for r in rewards]
        
        # 返回状态、奖励、是否终止、覆盖率
        next_state = self.get_state()
        return next_state, rewards, done, coverage

    def render(self, mode='human'):
        """渲染当前环境状态"""
        if mode == 'human':
            self.screen.fill(self.COLORS['background'])
            
            # 绘制地图
            for y in range(self.height):
                for x in range(self.width):
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                      self.cell_size, self.cell_size)
                    
                    if self.map[y][x] == 1:  # 障碍物
                        pygame.draw.rect(self.screen, self.COLORS['obstacle'], rect)
                    elif (x, y) in self.visited:  # 已访问区域
                        visit_count = self.visited[(x, y)]
                        # 颜色根据访问次数变化
                        color_intensity = min(255, 100 + min(visit_count, 5) * 30)
                        pygame.draw.rect(self.screen, (50, 150, color_intensity), rect)
                    
                    # 绘制网格
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
            
            # 绘制智能体
            for i, (x, y) in enumerate(self.agent_positions):
                center_x = x * self.cell_size + self.cell_size // 2
                center_y = y * self.cell_size + self.cell_size // 2
                color = self.COLORS['agent'][i % len(self.COLORS['agent'])]
                pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 3)
                pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), self.cell_size // 6)
                
                # 显示智能体ID
                font = pygame.font.SysFont(None, 20)
                text = font.render(str(i), True, (0, 0, 0))
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)
            
            # 显示统计信息
            font = pygame.font.SysFont(None, 24)
            total_cells = self.width * self.height - np.sum(self.map)
            coverage = len(self.unique_visits) / total_cells if total_cells > 0 else 0.0
            text = font.render(f"步数: {self.steps}/{self.max_steps} | 覆盖率: {coverage:.1%}", True, self.COLORS['text'])
            self.screen.blit(text, (10, 10))

            if True:
                pygame.image.save(self.screen, f"frames/{self.steps:04d}.png")
            
            pygame.display.flip()
            return self.screen
        else:
            return None

    def close(self):
        """关闭环境"""
        pygame.quit()


class PolicyNetwork(nn.Module):
    """策略网络 - 用于每个智能体的决策"""
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
    
    def act(self, state):
        """根据状态选择动作"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class MultiAgentPPO:
    """多智能体PPO算法实现"""
    def __init__(self, env, gamma=0.99, lr=0.001, clip_epsilon=0.2, update_interval=5):
        self.env = env
        self.gamma = gamma  # 折扣因子
        self.clip_epsilon = clip_epsilon  # PPO裁剪参数
        self.update_interval = update_interval  # 更新间隔
        
        # 为每个智能体创建策略网络
        self.policies = [PolicyNetwork(env.state_size, env.action_size) for _ in range(env.num_agents)]
        self.optimizers = [optim.Adam(policy.parameters(), lr=lr) for policy in self.policies]
        
        # 经验缓冲区
        self.buffer = [[] for _ in range(env.num_agents)]
        
        # 训练统计
        self.episode_rewards = []
        self.coverages = []
    
    def collect_experience(self, num_episodes):
        """收集经验并更新策略"""
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = [0] * self.env.num_agents
            log_probs = [[] for _ in range(self.env.num_agents)]
            rewards = [[] for _ in range(self.env.num_agents)]
            values = [[] for _ in range(self.env.num_agents)]
            
            done = False
            coverage = 0
            
            while not done:
                # 所有智能体选择动作
                actions = []
                for i in range(self.env.num_agents):
                    agent_state = state[i]
                    action, log_prob = self.policies[i].act(agent_state)
                    actions.append(action)
                    log_probs[i].append(log_prob)
                
                # 执行动作
                next_state, step_rewards, done, coverage = self.env.step(actions)
                
                # 存储经验
                for i in range(self.env.num_agents):
                    episode_reward[i] += step_rewards[i]
                    rewards[i].append(step_rewards[i])
                
                state = next_state
            
            # 记录训练统计
            self.episode_rewards.append(sum(episode_reward) / self.env.num_agents)  # 平均奖励
            self.coverages.append(coverage)
            
            # 更新策略
            self.update_policies(rewards, log_probs)
            
            # 定期打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / 10
                avg_coverage = sum(self.coverages[-10:]) / 10
                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Coverage: {avg_coverage:.1%}")
    
    def update_policies(self, rewards, log_probs):
        """使用PPO算法更新策略"""
        for i in range(self.env.num_agents):
            # 计算回报
            R = 0
            returns = []
            for r in reversed(rewards[i]):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # 计算损失
            log_probs_tensor = torch.stack(log_probs[i])
            loss = -log_probs_tensor * returns
            
            # 梯度下降
            self.optimizers[i].zero_grad()
            loss.mean().backward()
            self.optimizers[i].step()
    
    def save_models(self, path="patrol_models.pth"):
        """保存所有智能体模型"""
        state_dicts = [policy.state_dict() for policy in self.policies]
        torch.save(state_dicts, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path="patrol_models.pth"):
        """加载所有智能体模型"""
        state_dicts = torch.load(path)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(state_dicts[i])
        print(f"Models loaded from {path}")
    
    def plot_training(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Average Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.coverages)
        plt.title("Coverage per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Coverage")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("training_progress.png")
        plt.show()


def validate_model(env, policies, num_episodes=3, render=True, delay=0.05):
    """
    验证训练好的模型
    
    参数:
    env: 巡逻环境
    policies: 策略网络列表
    num_episodes: 验证的回合数
    render: 是否可视化
    delay: 渲染延迟
    """
    coverages = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_coverage = []
        
        while not done:
            # 所有智能体选择动作
            actions = []
            for i in range(env.num_agents):
                agent_state = state[i]
                with torch.no_grad():
                    action, _ = policies[i].act(agent_state)
                actions.append(action)
            
            # 执行动作
            state, _, done, coverage = env.step(actions)
            episode_coverage.append(coverage)
            
            # 渲染
            if render:
                env.render()
                time.sleep(delay)
        
        # 记录覆盖率
        coverages.append(max(episode_coverage))
        print(f"Episode {episode+1}: Max Coverage = {max(episode_coverage):.1%}")
    
    # 输出平均覆盖率
    avg_coverage = sum(coverages) / len(coverages)
    print(f"\nAverage Max Coverage: {avg_coverage:.1%}")
    return avg_coverage


def train_patrol_agents():
    """训练多智能体巡逻策略"""
    # 创建环境
    env = MultiAgentPatrolEnv(map_size=(15, 15), num_agents=4, max_steps=300)
    
    # 创建PPO训练器
    trainer = MultiAgentPPO(env, lr=0.0003)
    
    # 训练模型
    print("开始训练巡逻智能体...")
    start_time = time.time()
    trainer.collect_experience(num_episodes=200)
    training_time = time.time() - start_time
    print(f"训练完成! 用时: {training_time:.1f}秒")
    
    # 保存模型
    trainer.save_models()
    
    # 绘制训练曲线
    # trainer.plot_training()
    
    # 验证训练好的模型
    print("\n验证训练好的模型:")
    validate_model(env, trainer.policies)
    
    env.close()


if __name__ == "__main__":
    # 训练或加载预训练模型
    # train_patrol_agents()
    
    # 如果已有训练好的模型，可以直接加载验证
    env = MultiAgentPatrolEnv(map_size=(20, 20), num_agents=4)
    policies = [PolicyNetwork(env.state_size, env.action_size) for _ in range(env.num_agents)]
    trainer = MultiAgentPPO(env)
    trainer.load_models()
    validate_model(env, trainer.policies, num_episodes=1)