import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import random
import os
from collections import deque
import copy
import time
import pygame
from typing import Any, Dict, List, Optional, Tuple


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 环境参数
WORLD_SIZE = 10.0  # 世界大小
CAPTURE_DISTANCE = 0.5  # 抓捕距离
MAX_STEPS = 200  # 最大步数

# MADDPG参数
NUM_PURSUERS = 3
NUM_TARGETS = 3
STATE_DIM = 2  # 每个智能体的状态维度 (x, y)
ACTION_DIM = 2  # 每个智能体的动作维度 (dx, dy)
HIDDEN_DIM = 128
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GAMMA = 0.95
TAU = 1e-3
BUFFER_SIZE = 100000
BATCH_SIZE = 256
MAX_EPISODES = 2000
MAX_EPISODE_STEPS = MAX_STEPS
NOISE_DECAY = 0.9995
INITIAL_NOISE = 1.0
MIN_NOISE = 0.01

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境类
class PursuitEnv:
    def __init__(self):
        self.reset()

        self.screen = None
        self.screen_size = 600
        self.arena_size = WORLD_SIZE
        self.scale = self.screen_size / (self.arena_size * 2)
        self.capture_radius = CAPTURE_DISTANCE
        self.captures = np.zeros(NUM_TARGETS)
    
    def reset(self):
        # 随机初始化抓捕者和目标位置
        self.pursuers = np.random.uniform(0, WORLD_SIZE, (NUM_PURSUERS, 2))
        self.targets = np.random.uniform(0, WORLD_SIZE, (NUM_TARGETS, 2))
        self.steps = 0
        self.captured = [False] * NUM_TARGETS
        return self.get_state()
    
    def get_state(self):
        # 状态包含所有智能体的位置
        state = np.concatenate([self.pursuers.flatten(), self.targets.flatten()])
        return state
    
    def step(self, pursuer_actions, target_actions):
        # 更新抓捕者位置
        for i in range(NUM_PURSUERS):
            action = pursuer_actions[i]
            # 动作限制
            action = np.clip(action, -1.0, 1.0) * 0.2
            self.pursuers[i] += action
            # 边界限制
            self.pursuers[i] = np.clip(self.pursuers[i], 0, WORLD_SIZE)
        
        # 更新目标位置
        for i in range(NUM_TARGETS):
            if not self.captured[i]:
                action = target_actions[i]
                # 动作限制
                action = np.clip(action, -1.0, 1.0) * 0.15
                self.targets[i] += action
                # 边界限制
                self.targets[i] = np.clip(self.targets[i], 0, WORLD_SIZE)
        
        # 检查抓捕
        for i in range(NUM_TARGETS):
            if not self.captured[i]:
                for j in range(NUM_PURSUERS):
                    dist = np.linalg.norm(self.pursuers[j] - self.targets[i])
                    if dist < CAPTURE_DISTANCE:
                        self.captured[i] = True
                        break
        
        # 计算奖励
        pursuer_rewards = np.zeros(NUM_PURSUERS)
        target_rewards = np.zeros(NUM_TARGETS)
        
        # 抓捕者奖励：鼓励抓捕目标
        for i in range(NUM_PURSUERS):
            min_dist = min([np.linalg.norm(self.pursuers[i] - self.targets[j]) 
                           for j in range(NUM_TARGETS) if not self.captured[j]], default=WORLD_SIZE)
            pursuer_rewards[i] = -min_dist * 0.1  # 鼓励接近目标
            
            # 额外奖励：如果抓捕到目标
            for j in range(NUM_TARGETS):
                if not self.captured[j]:
                    dist = np.linalg.norm(self.pursuers[i] - self.targets[j])
                    if dist < CAPTURE_DISTANCE:
                        pursuer_rewards[i] += 10.0
        
        # 目标奖励：鼓励躲避抓捕者
        for i in range(NUM_TARGETS):
            if not self.captured[i]:
                min_dist = min([np.linalg.norm(self.targets[i] - self.pursuers[j]) 
                             for j in range(NUM_PURSUERS)])
                target_rewards[i] = min_dist * 0.1  # 鼓励远离抓捕者
            else:
                target_rewards[i] = -10.0  # 被抓住的惩罚
        
        # 步数惩罚
        pursuer_rewards -= 0.01
        target_rewards -= 0.01
        
        # 检查终止条件
        self.steps += 1
        done = (self.steps >= MAX_STEPS) or all(self.captured)
        
        return self.get_state(), pursuer_rewards, target_rewards, done, self.captured
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('HG title')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))

        border_rect = pygame.Rect(0, 0, self.screen_size, self.screen_size)

        pygame.draw.rect(self.screen, (0, 0, 0), border_rect, 2)

        for pos in self.pursuers:
            pygame.draw.circle(self.screen, (0, 0, 255), self._scale_position(pos), 10)
            pygame.draw.circle(self.screen, (200, 200, 200), self._scale_position(pos), int(self.capture_radius * self.scale), 1)

        for i, pos in enumerate(self.targets):
            if self.captures[i] == 0:
                pygame.draw.circle(self.screen, (255, 0, 0), self._scale_position(pos), 8)
            else:
                pygame.draw.circle(self.screen, (255, 255, 0), self._scale_position(pos), 8)
        pygame.display.flip()

        self.clock.tick(30)

    def _scale_position(self, pos: np.ndarray) -> Tuple[int, int]:
        x = (pos[0] + self.arena_size) * self.scale
        y = (self.arena_size - pos[1]) * self.scale
        return (int(x), int(y))

from ActorCritic import *

# Actor网络
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         for layer in [self.fc1, self.fc2, self.fc3]:
#             nn.init.xavier_uniform_(layer.weight)
#             nn.init.constant_(layer.bias, 0.1)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.tanh(self.fc3(x))  # 输出在[-1, 1]范围内
#         return x

# # Critic网络 (修复维度问题)
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
#         super(Critic, self).__init__()
#         # 输入状态和所有智能体的动作
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         total_dim = state_dim + action_dim
        
#         self.fc1 = nn.Linear(total_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         for layer in [self.fc1, self.fc2, self.fc3]:
#             nn.init.xavier_uniform_(layer.weight)
#             nn.init.constant_(layer.bias, 0.1)
    
#     def forward(self, state, actions):
#         # 拼接状态和所有动作
#         actions = torch.cat(actions, dim=1)
#         x = torch.cat([state, actions], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# MADDPG Agent
class MADDPGAgent:
    def __init__(self, actor_state_dim, action_dim, is_pursuer=True):
        self.is_pursuer = is_pursuer
        self.actor = Actor(actor_state_dim, action_dim).to(device)
        self.target_actor = Actor(actor_state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        self.noise = INITIAL_NOISE
    
    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()
        
        if noise:
            # 添加探索噪声
            action += self.noise * np.random.normal(0, 0.1, size=ACTION_DIM)
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def update_noise(self):
        self.noise = max(MIN_NOISE, self.noise * NOISE_DECAY)

# MADDPG 训练器
class MADDPGTrainer:
    def __init__(self):
        # 经验回放
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # 创建抓捕者和目标智能体
        total_state_dim = STATE_DIM * (NUM_PURSUERS + NUM_TARGETS)
        
        # 抓捕者Actor
        self.pursuer_agents = [
            MADDPGAgent(total_state_dim, ACTION_DIM, is_pursuer=True)
            for _ in range(NUM_PURSUERS)
        ]
        
        # 目标Actor
        self.target_agents = [
            MADDPGAgent(total_state_dim, ACTION_DIM, is_pursuer=False)
            for _ in range(NUM_TARGETS)
        ]
        
        # 抓捕者Critic
        self.pursuer_critic = Critic(
            total_state_dim, 
            ACTION_DIM * NUM_PURSUERS
        ).to(device)
        self.target_pursuer_critic = Critic(
            total_state_dim, 
            ACTION_DIM * NUM_PURSUERS
        ).to(device)
        self.target_pursuer_critic.load_state_dict(self.pursuer_critic.state_dict())
        self.pursuer_critic_optimizer = optim.Adam(self.pursuer_critic.parameters(), lr=CRITIC_LR)
        
        # 目标Critic
        self.target_critic = Critic(
            total_state_dim, 
            ACTION_DIM * NUM_TARGETS
        ).to(device)
        self.target_target_critic = Critic(
            total_state_dim, 
            ACTION_DIM * NUM_TARGETS
        ).to(device)
        self.target_target_critic.load_state_dict(self.target_critic.state_dict())
        self.target_critic_optimizer = optim.Adam(self.target_critic.parameters(), lr=CRITIC_LR)
        
        # 环境
        self.env = PursuitEnv()
        
        # 训练记录
        self.episode_rewards = []
        self.capture_rates = []
    
    def save_experience(self, state, pursuer_actions, target_actions, 
                       pursuer_rewards, target_rewards, next_state, done):
        self.buffer.append((
            state, pursuer_actions, target_actions, 
            pursuer_rewards, target_rewards, next_state, done
        ))
    
    def sample_batch(self):
        if len(self.buffer) < BATCH_SIZE:
            return None
        
        batch = random.sample(self.buffer, BATCH_SIZE)
        
        # 解包批数据
        states, pursuer_actions, target_actions, pursuer_rewards, target_rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(device)
        pursuer_actions = torch.FloatTensor(np.array(pursuer_actions)).to(device)
        target_actions = torch.FloatTensor(np.array(target_actions)).to(device)
        pursuer_rewards = torch.FloatTensor(np.array(pursuer_rewards)).to(device)
        target_rewards = torch.FloatTensor(np.array(target_rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        
        return states, pursuer_actions, target_actions, pursuer_rewards, target_rewards, next_states, dones
    
    def update(self):
        batch = self.sample_batch()
        if batch is None:
            return 0, 0, 0, 0
        
        states, pursuer_actions, target_actions, pursuer_rewards, target_rewards, next_states, dones = batch
        
        # 更新抓捕者Critic
        with torch.no_grad():
            # 计算目标Q值
            next_pursuer_actions = []
            for agent in self.pursuer_agents:
                next_pursuer_actions.append(agent.target_actor(next_states))
            
            next_target_actions = []
            for agent in self.target_agents:
                next_target_actions.append(agent.target_actor(next_states))
            
            # 抓捕者目标Q值
            pursuer_q_next = self.target_pursuer_critic(next_states, next_pursuer_actions)
            pursuer_q_target = pursuer_rewards + GAMMA * (1 - dones) * pursuer_q_next
            
            # 目标目标Q值
            target_q_next = self.target_target_critic(next_states, next_target_actions)
            target_q_target = target_rewards + GAMMA * (1 - dones) * target_q_next
        
        # 抓捕者Critic损失
        current_pursuer_q = self.pursuer_critic(states, [
            pursuer_actions[:, i*ACTION_DIM:(i+1)*ACTION_DIM] for i in range(NUM_PURSUERS)
        ])
        pursuer_critic_loss = F.mse_loss(current_pursuer_q, pursuer_q_target.detach())
        
        # 目标Critic损失
        current_target_q = self.target_critic(states, [
            target_actions[:, i*ACTION_DIM:(i+1)*ACTION_DIM] for i in range(NUM_TARGETS)
        ])
        target_critic_loss = F.mse_loss(current_target_q, target_q_target.detach())
        
        # 优化Critic
        self.pursuer_critic_optimizer.zero_grad()
        pursuer_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pursuer_critic.parameters(), 1.0)
        self.pursuer_critic_optimizer.step()
        
        self.target_critic_optimizer.zero_grad()
        target_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.target_critic.parameters(), 1.0)
        self.target_critic_optimizer.step()
        
        # 更新Actor
        pursuer_actor_loss = 0
        for i, agent in enumerate(self.pursuer_agents):
            # 重新计算动作
            new_pursuer_actions = []
            for j, a in enumerate(self.pursuer_agents):
                if j == i:
                    new_pursuer_actions.append(agent.actor(states))
                else:
                    # 使用当前动作
                    new_pursuer_actions.append(
                        pursuer_actions[:, j*ACTION_DIM:(j+1)*ACTION_DIM].detach()
                    )
            
            # 计算Actor损失
            actor_loss = -self.pursuer_critic(states, new_pursuer_actions).mean()
            pursuer_actor_loss += actor_loss
            
            # 优化Actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
        
        target_actor_loss = 0
        for i, agent in enumerate(self.target_agents):
            # 重新计算动作
            new_target_actions = []
            for j, a in enumerate(self.target_agents):
                if j == i:
                    new_target_actions.append(agent.actor(states))
                else:
                    # 使用当前动作
                    new_target_actions.append(
                        target_actions[:, j*ACTION_DIM:(j+1)*ACTION_DIM].detach()
                    )
            
            # 计算Actor损失
            actor_loss = -self.target_critic(states, new_target_actions).mean()
            target_actor_loss += actor_loss
            
            # 优化Actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
        
        # 更新目标网络
        for agent in self.pursuer_agents:
            for param, target_param in zip(agent.actor.parameters(), agent.target_actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for agent in self.target_agents:
            for param, target_param in zip(agent.actor.parameters(), agent.target_actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for param, target_param in zip(self.pursuer_critic.parameters(), self.target_pursuer_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for param, target_param in zip(self.target_critic.parameters(), self.target_target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return pursuer_critic_loss.item(), target_critic_loss.item(), pursuer_actor_loss.item(), target_actor_loss.item()
    
    def train(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        
        for episode in range(MAX_EPISODES):
            state = self.env.reset()
            episode_pursuer_reward = np.zeros(NUM_PURSUERS)
            episode_target_reward = np.zeros(NUM_TARGETS)
            captured_count = 0
            
            for step in range(MAX_EPISODE_STEPS):
                # 选择动作
                pursuer_actions = []
                for agent in self.pursuer_agents:
                    action = agent.select_action(state)
                    pursuer_actions.append(action)
                
                target_actions = []
                for agent in self.target_agents:
                    action = agent.select_action(state)
                    target_actions.append(action)
                
                # 执行动作
                next_state, pursuer_rewards, target_rewards, done, captured = self.env.step(
                    np.array(pursuer_actions), np.array(target_actions)
                )
                
                # 存储经验
                self.save_experience(
                    state, 
                    np.concatenate(pursuer_actions), 
                    np.concatenate(target_actions), 
                    np.array(pursuer_rewards), 
                    np.array(target_rewards), 
                    next_state, 
                    done
                )
                
                # 更新奖励
                episode_pursuer_reward += pursuer_rewards
                episode_target_reward += target_rewards
                captured_count += sum(captured)
                
                # 更新状态
                state = next_state
                
                # 更新模型
                losses = self.update()
                
                # 更新噪声
                for agent in self.pursuer_agents:
                    agent.update_noise()
                for agent in self.target_agents:
                    agent.update_noise()
                
                if done:
                    break
            
            # 记录结果
            total_reward = np.sum(episode_pursuer_reward) + np.sum(episode_target_reward)
            self.episode_rewards.append(total_reward)
            capture_rate = captured_count / NUM_TARGETS
            self.capture_rates.append(capture_rate)
            
            # 打印进度
            if episode % 50 == 0 or episode == MAX_EPISODES - 1:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else total_reward
                avg_capture = np.mean(self.capture_rates[-50:]) if len(self.capture_rates) >= 50 else capture_rate
                print(f"Episode {episode:4d} | Total Reward: {total_reward:7.2f} | "
                      f"Avg Reward: {avg_reward:7.2f} | Capture Rate: {capture_rate:.2f} | "
                      f"Avg Capture: {avg_capture:.2f} | Noise: {self.pursuer_agents[0].noise:.4f}")
            
            # 定期保存模型
            if episode % 200 == 0 or episode == MAX_EPISODES - 1:
                self.save_models(episode)
        
    def save_models(self, episode):
        # 保存抓捕者模型
        for i, agent in enumerate(self.pursuer_agents):
            torch.save(agent.actor.state_dict(), f"models/pursuer_actor_{i}_ep{episode}.pth")
            torch.save(agent.target_actor.state_dict(), f"models/pursuer_target_actor_{i}_ep{episode}.pth")
        
        # 保存目标模型
        for i, agent in enumerate(self.target_agents):
            torch.save(agent.actor.state_dict(), f"models/target_actor_{i}_ep{episode}.pth")
            torch.save(agent.target_actor.state_dict(), f"models/target_target_actor_{i}_ep{episode}.pth")
        
        # 保存Critic模型
        torch.save(self.pursuer_critic.state_dict(), f"models/pursuer_critic_ep{episode}.pth")
        torch.save(self.target_pursuer_critic.state_dict(), f"models/pursuer_target_critic_ep{episode}.pth")
        torch.save(self.target_critic.state_dict(), f"models/target_critic_ep{episode}.pth")
        torch.save(self.target_target_critic.state_dict(), f"models/target_target_critic_ep{episode}.pth")
        
        print(f"Models saved at episode {episode}")
    
    def load_models(self, episode):
        # 加载抓捕者模型
        for i, agent in enumerate(self.pursuer_agents):
            agent.actor.load_state_dict(torch.load(f"models/pursuer_actor_{i}_ep{episode}.pth", map_location=device))
            agent.target_actor.load_state_dict(torch.load(f"models/pursuer_target_actor_{i}_ep{episode}.pth", map_location=device))
        
        # 加载目标模型
        for i, agent in enumerate(self.target_agents):
            agent.actor.load_state_dict(torch.load(f"models/target_actor_{i}_ep{episode}.pth", map_location=device))
            agent.target_actor.load_state_dict(torch.load(f"models/target_target_actor_{i}_ep{episode}.pth", map_location=device))
        
        # 加载Critic模型
        self.pursuer_critic.load_state_dict(torch.load(f"models/pursuer_critic_ep{episode}.pth", map_location=device))
        self.target_pursuer_critic.load_state_dict(torch.load(f"models/pursuer_target_critic_ep{episode}.pth", map_location=device))
        self.target_critic.load_state_dict(torch.load(f"models/target_critic_ep{episode}.pth", map_location=device))
        self.target_target_critic.load_state_dict(torch.load(f"models/target_target_critic_ep{episode}.pth", map_location=device))
        
        print(f"Models loaded from episode {episode}")
    
    def validate(self, episode, num_episodes=3, render=True):
        # 加载模型
        self.load_models(episode)
        
        for ep in range(num_episodes):
            state = self.env.reset()
            total_pursuer_reward = np.zeros(NUM_PURSUERS)
            total_target_reward = np.zeros(NUM_TARGETS)
            captured_count = 0
            step_count = 0
            
            for step in range(MAX_EPISODE_STEPS):
                # 选择动作（无噪声）
                pursuer_actions = []
                for agent in self.pursuer_agents:
                    action = agent.select_action(state, noise=False)
                    pursuer_actions.append(action)
                
                target_actions = []
                for agent in self.target_agents:
                    action = agent.select_action(state, noise=False)
                    target_actions.append(action)
                
                # 执行动作
                next_state, pursuer_rewards, target_rewards, done, captured = self.env.step(
                    np.array(pursuer_actions), np.array(target_actions))
                
                # 更新奖励
                total_pursuer_reward += pursuer_rewards
                total_target_reward += target_rewards
                captured_count += sum(captured)
                
                # 渲染
                if render:
                    self.env.render()
                    time.sleep(0.05)
                
                # 更新状态
                state = next_state
                step_count = step + 1
                
                if done:
                    break
            
            # 打印结果
            total_reward = np.sum(total_pursuer_reward) + np.sum(total_target_reward)
            capture_rate = captured_count / NUM_TARGETS
            print(f"Validation Episode {ep} | Total Reward: {total_reward:.2f} | "
                  f"Capture Rate: {capture_rate:.2f} | Steps: {step_count}")

if __name__ == "__main__":
    trainer = MADDPGTrainer()
    
    # 训练模型
    # trainer.train()
    
    # 验证模型
    trainer.validate(episode=MAX_EPISODES-1, num_episodes=3, render=True)