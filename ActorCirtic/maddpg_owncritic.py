'''
自定义environment和Actor-critic网络写一个多智能体抓捕MADDPG,要求如下：
1. 有3个抓捕者和3个目标
2. action连续
3. 抓捕者和目标分离训练，但共享环境
4. 存储训练模型到文件
5. 能够有一个函数验证训练好的模型，验证时有实时render可视化抓捕过程
'''

import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from BaseEnv import BaseEnv
from ActorCritic import *

# 环境参数
NUM_PURSUERS = 3
NUM_TARGETS = 2
WORLD_SIZE = 10.0
CAPTURE_DIST = 0.5
TIME_LIMIT = 200

# MADDPG参数
ACTOR_LR = 0.0005
CRITIC_LR = 0.001
GAMMA = 0.95
TAU = 0.01
BUFFER_SIZE = 100000
BATCH_SIZE = 512
NOISE_DECAY = 0.9995
MAX_EPISODES = 5000  # 增加训练回合数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiAgentEnv(BaseEnv):
    def reset(self, seed = (int)(time.time())):
        super().reset(seed = seed)
        return self._get_obs()

    def _get_obs(self):
        state = np.concatenate([self.pursuers.flatten(), self.targets.flatten()])
        return state

    def step(self, pursuer_actions, target_actions):
        for i in range(self.num_pursuers):
            action = pursuer_actions[i]
            action = np.clip(action, -1.0, 1.0) * 0.2
            self.pursuers[i] += action
            self.pursuers[i] = np.clip(self.pursuers[i], 0, self.arena_size)

        for j in range(self.num_targets):
            if self.captures[j] == 0:
                action = target_actions[j]
                action = np.clip(action, -1.0, 1.0) * 0.15
                self.targets[j] += action
                self.targets[j] = np.clip(self.targets[j], 0, self.arena_size)
        
        pursuer_rewards = np.zeros(self.num_pursuers)
        target_rewards = np.zeros(self.num_targets)

        # capture
        for j in range(self.num_targets):
            if self.captures[j] != 0:
                continue
            for i in range(self.num_pursuers):
                dist = np.linalg.norm(self.pursuers[i] - self.targets[j])
                if dist < self.capture_radius:
                    if self.captures[j] == 0:
                        self.captures[j] = 1
                        target_rewards[j] -= 10.0
                    pursuer_rewards[i] += 10.0 

        # pursuers dist
        for i in range(self.num_pursuers):
            min_dist = min([np.linalg.norm(self.pursuers[i] - self.targets[j]) 
                            for j in range(self.num_targets) if not self.captures[j]], default = self.arena_size)
            pursuer_rewards[i] = -min_dist * 0.1

        for j in range(self.num_targets):
            if self.captures[j] == 0:
                min_dist = min([np.linalg.norm(self.pursuers[i] - self.targets[j]) 
                            for i in range(self.num_pursuers)])
                target_rewards[j] += min_dist * 0.1
            dist_to_boundary = min(np.abs(self.targets[j][0]), 
                                np.abs(self.targets[j][1]), 
                                np.abs(self.arena_size - np.abs(self.targets[j][0])), 
                                np.abs(self.arena_size - np.abs(self.targets[j][1]))
                                )
            if dist_to_boundary < 2.0:
                target_rewards[j] -= dist_to_boundary * 0.1

        pursuer_rewards -= 0.01
        target_rewards -= 0.01

        self.current_step += 1
        done = (self.current_step >= self.max_steps) or all(self.captures != 0)
        return self._get_obs(), pursuer_rewards, target_rewards, done, self.captures

    def get_positions(self):
        return self.pursuers.copy(), self.targets.copy(), self.captures.copy()

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, total_action_dim, is_pursuer=True):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, total_action_dim).to(device)
        self.target_critic = Critic(state_dim, total_action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        self.is_pursuer = is_pursuer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.noise_scale = 1.0
        self.ou_state = np.zeros(action_dim)  # 初始化OU状态
        self.ou_theta = 0.2  # OU过程参数
        self.ou_sigma = 0.3  # OU过程参数
    
    def reset_noise(self):
        self.ou_state = np.zeros(self.action_dim)
    
    def decay_noise(self):
        self.noise_scale = max(0.1, self.noise_scale * NOISE_DECAY)
    
    def act(self, state, noise=True):
        # 设置网络为评估模式以处理单个样本
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()  # 恢复训练模式
        
        if noise:
            noise_val = self.ou_noise() * self.noise_scale
            action = np.clip(action + noise_val, -1.0, 1.0)
        return action
    
    def ou_noise(self):
        """Ornstein-Uhlenbeck过程生成噪声"""
        noise = np.random.normal(0, self.ou_sigma, self.action_dim)
        self.ou_state = self.ou_state + self.ou_theta * (-self.ou_state) + noise
        return self.ou_state
    
    def update(self, replay_buffer, agents, agent_idx):
        # 确保使用批量数据更新
        states, all_actions, rewards, next_states, dones = replay_buffer.sample()
        
        # 更新Critic
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(agents):
                next_actions.append(agent.target_actor(next_states))
            next_actions = torch.cat(next_actions, dim=1)
            
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards[:, agent_idx].unsqueeze(1) + GAMMA * (1 - dones[:, agent_idx].unsqueeze(1)) * target_q
        
        current_q = self.critic(states, all_actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # 更新Actor
        for agent in agents:
            agent.actor.requires_grad_(False)
        self.actor.requires_grad_(True)
        
        current_action = self.actor(states)
        other_actions = []
        for i, agent in enumerate(agents):
            if i != agent_idx:
                other_actions.append(all_actions[:, i*self.action_dim:(i+1)*self.action_dim])
        
        if other_actions:
            other_actions = torch.cat(other_actions, dim=1)
            actor_actions = torch.cat([all_actions[:, :agent_idx*self.action_dim], 
                                      current_action,
                                      all_actions[:, (agent_idx+1)*self.action_dim:]], dim=1)
        else:
            actor_actions = current_action
        
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        for agent in agents:
            agent.actor.requires_grad_(True)
        
        # 更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, num_agents):
        self.states = np.zeros((BUFFER_SIZE, state_dim))
        self.actions = np.zeros((BUFFER_SIZE, num_agents * action_dim))
        self.rewards = np.zeros((BUFFER_SIZE, num_agents))
        self.next_states = np.zeros((BUFFER_SIZE, state_dim))
        self.dones = np.zeros((BUFFER_SIZE, num_agents))
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
    
    def add(self, state, actions, rewards, next_state, dones):
        self.states[self.ptr] = state
        self.actions[self.ptr] = np.concatenate(actions)
        self.rewards[self.ptr] = rewards
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = dones
        self.ptr = (self.ptr + 1) % BUFFER_SIZE
        self.size = min(self.size + 1, BUFFER_SIZE)
    
    def sample(self):
        if self.size < BATCH_SIZE:
            return None, None, None, None, None
        
        idx = np.random.choice(self.size, BATCH_SIZE, replace=False)
        states = torch.FloatTensor(self.states[idx]).to(device)
        actions = torch.FloatTensor(self.actions[idx]).to(device)
        rewards = torch.FloatTensor(self.rewards[idx]).to(device)
        next_states = torch.FloatTensor(self.next_states[idx]).to(device)
        dones = torch.FloatTensor(self.dones[idx]).to(device)
        return states, actions, rewards, next_states, dones

def train_maddpg(episodes=MAX_EPISODES, save_dir="maddpg_model"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    env = MultiAgentEnv()
    state_dim = len(env._get_obs())
    action_dim = 2
    num_agents = NUM_PURSUERS + NUM_TARGETS
    total_action_dim = num_agents * action_dim
    
    agents = []
    for i in range(num_agents):
        is_pursuer = (i < NUM_PURSUERS)
        agents.append(MADDPGAgent(state_dim, action_dim, total_action_dim, is_pursuer))
        agents[-1].reset_noise()
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, num_agents)
    
    # 训练统计
    capture_rates = []
    pursuer_rewards = []
    target_rewards = []
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = np.zeros(num_agents)
        episode_captured = 0
        
        # 重置所有智能体的噪声状态
        for agent in agents:
            agent.reset_noise()
        
        # 衰减噪声
        if episode % 100 == 0:
            for agent in agents:
                agent.decay_noise()
        
        step_count = 0
        while True:
            actions = []
            for i, agent in enumerate(agents):
                # 抓捕者使用更多噪声
                noise_factor = 1.0 if i < NUM_PURSUERS else 0.8
                action = agent.act(state, noise=True) * noise_factor
                actions.append(action)
            
            pursuer_actions = np.array(actions[:NUM_PURSUERS])
            target_actions = np.array(actions[NUM_PURSUERS:])
            
            next_state, pursuer_reward, target_reward, done, captured = env.step(
                pursuer_actions, target_actions
            )
            
            # rewards = np.concatenate([
            #     [pursuer_reward] * NUM_PURSUERS,
            #     [target_reward] * NUM_TARGETS
            # ])
            rewards = np.concatenate([pursuer_reward, target_reward])
            
            dones = np.concatenate([
                [done] * NUM_PURSUERS,
                [done] * NUM_TARGETS
            ])
            
            replay_buffer.add(state, actions, rewards, next_state, dones)
            
            state = next_state
            episode_rewards += rewards
            episode_captured = sum(captured)
            step_count += 1
            
            if done:
                break
        
        # 更新智能体
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        if states is not None:
            critic_losses, actor_losses = [], []
            for i, agent in enumerate(agents):
                critic_loss, actor_loss = agent.update(replay_buffer, agents, i)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
        
        # 记录统计
        capture_rates.append(episode_captured / NUM_TARGETS)
        pursuer_rewards.append(episode_rewards[:NUM_PURSUERS].mean())
        target_rewards.append(episode_rewards[NUM_PURSUERS:].mean())
        
        # 打印进度
        if episode % 100 == 0:
            pursuer_reward = pursuer_rewards[-1]
            target_reward = target_rewards[-1]
            elapsed = time.time() - start_time
            time_per_episode = elapsed / (episode + 1) if episode > 0 else 0
            
            print(f"Episode {episode}/{episodes} | "
                  f"Pursuer Reward: {pursuer_reward:.2f} | "
                  f"Target Reward: {target_reward:.2f} | "
                  f"Captured: {episode_captured}/{NUM_TARGETS} | "
                  f"Noise: {agents[0].noise_scale:.3f} | "
                  f"Time: {elapsed:.1f}s ({time_per_episode:.3f}s/ep)")
            
            if states is not None:
                avg_critic_loss = np.mean(critic_losses)
                avg_actor_loss = np.mean(actor_losses)
                print(f"  Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
        
        # 每500回合保存一次模型
        if episode % 500 == 0 and episode > 0:
            for i, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"agent_{i}_ep{episode}.pth"))
    
    # 保存最终模型
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"agent_{i}_final.pth"))
    
    print(f"Training completed in {time.time()-start_time:.1f} seconds. Models saved to {save_dir}")

def validate_model(model_dir="maddpg_model", episodes=5, model_suffix="final"):
    env = MultiAgentEnv(render_mode = ['human'])
    state_dim = len(env._get_obs())
    action_dim = 2
    num_agents = NUM_PURSUERS + NUM_TARGETS
    total_action_dim = num_agents * action_dim
    
    agents = []
    for i in range(num_agents):
        is_pursuer = (i < NUM_PURSUERS)
        agent = MADDPGAgent(state_dim, action_dim, total_action_dim, is_pursuer)
        model_path = os.path.join(model_dir, f"agent_{i}_{model_suffix}.pth")
        agent.actor.load_state_dict(torch.load(model_path, map_location=device))
        agent.actor.eval()
        agents.append(agent)
    
    capture_counts = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        
        total_pursuer_reward = 0
        total_target_reward = 0
        captured_count = 0
        step = 0
        
        def update(frame = None):
            nonlocal state, total_pursuer_reward, total_target_reward, captured_count, step
            
            actions = []
            for agent in agents:
                action = agent.act(state, noise=False)
                actions.append(action)
            
            pursuer_actions = np.array(actions[:NUM_PURSUERS])
            target_actions = np.array(actions[NUM_PURSUERS:])
            
            next_state, pursuer_reward, target_reward, done, captured = env.step(
                pursuer_actions, target_actions
            )
            state = next_state
            
            step += 1
            
            pursuers, targets, captured_flags = env.get_positions()
            
            return done
        
        done = False
        while done != True:
            done = update()
            env.render()
            time.sleep(0.5)

if __name__ == "__main__":
    # 训练模型
    train_maddpg(episodes=MAX_EPISODES)
    
    # 验证模型
    validate_model(episodes=3)