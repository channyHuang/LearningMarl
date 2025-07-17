'''
自定义environment和写一个多智能体抓捕PPO,要求如下：
1. 有多个抓捕者和多个目标
2. 抓捕者尽可能抓捕，目标尽可能逃跑，action连续
3. 存储训练模型到文件
4. 能够有一个函数验证训练好的模型，验证时有实时render可视化抓捕过程
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from typing import Any, Dict, List, Optional, Tuple

class EnhancedHuntEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, num_pursuers=3, num_targets=2, arena_size=10.0, 
                 capture_radius=1.0, max_steps=300, render_mode=None):
        super().__init__()
        
        # 环境参数调整
        self.num_pursuers = num_pursuers
        self.num_targets = num_targets
        self.arena_size = arena_size
        self.capture_radius = capture_radius  # 增大捕获半径
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # 动作空间：标准化到[-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(num_pursuers * 2,), 
            dtype=np.float32
        )
        
        # 改进的观察空间：包含相对位置信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_pursuers * num_targets * 2 + num_pursuers * 2,), 
            dtype=np.float32
        )
        
        # 初始化
        self.pursuers = np.zeros((num_pursuers, 2), dtype=np.float32)
        self.targets = np.zeros((num_targets, 2), dtype=np.float32)
        self.current_step = 0
        
        # 渲染设置
        self.screen = None
        self.clock = None
        self.screen_size = 600
        self.scale = self.screen_size / (arena_size * 2)
        
    def reset(self, seed=None, options=None):
        # 重置状态
        super().reset(seed=seed)
        self.current_step = 0
        
        # 追捕者在左侧半圆
        angles = np.linspace(0, np.pi, self.num_pursuers)
        self.pursuers[:, 0] = -self.arena_size * 0.5 * np.cos(angles)
        self.pursuers[:, 1] = self.arena_size * 0.5 * np.sin(angles)
        
        # 目标在右侧随机分布
        self.targets[:, 0] = self.np_random.uniform(0, self.arena_size*0.8, size=self.num_targets)
        self.targets[:, 1] = self.np_random.uniform(-self.arena_size*0.8, self.arena_size*0.8, size=self.num_targets)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # 改进的观察值：包含相对位置和自身位置
        obs = []
        for pursuer in self.pursuers:
            for target in self.targets:
                obs.extend(target - pursuer)  # 相对位置
        obs.extend(self.pursuers.flatten())  # 自身位置
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        self.current_step += 1
        
        # 解析动作
        actions = actions.reshape((self.num_pursuers, 2))
        
        # 更新追捕者位置（加入动量）
        self.pursuers += actions * 0.25  # 降低移动速度
        
        # 目标逃跑策略（更智能）
        for i in range(self.num_targets):
            closest_dist = np.inf
            escape_vec = np.zeros(2)
            
            # 计算来自所有追捕者的威胁
            for pursuer in self.pursuers:
                dist = np.linalg.norm(pursuer - self.targets[i])
                if dist < closest_dist:
                    closest_dist = dist
                    escape_vec = self.targets[i] - pursuer
            
            if closest_dist < self.arena_size * 0.3:  # 只在危险时逃跑
                if np.linalg.norm(escape_vec) > 0:
                    escape_vec = escape_vec / np.linalg.norm(escape_vec)
                self.targets[i] += escape_vec * 0.3  # 目标速度
        
        # 边界检查
        self.pursuers = np.clip(self.pursuers, -self.arena_size, self.arena_size)
        self.targets = np.clip(self.targets, -self.arena_size, self.arena_size)
        
        # 计算奖励（关键改进）
        rewards = np.zeros(self.num_pursuers)
        captures = 0
        
        for i in range(self.num_targets):
            min_dist = np.inf
            for j in range(self.num_pursuers):
                dist = np.linalg.norm(self.pursuers[j] - self.targets[i])
                if dist < self.capture_radius:
                    captures += 1
                    rewards[j] += 10.0  # 个体捕获奖励
                    self.targets[i] = [  # 重置目标
                        self.np_random.uniform(0, self.arena_size*0.8),
                        self.np_random.uniform(-self.arena_size*0.8, self.arena_size*0.8)
                    ]
                    break
                elif dist < min_dist:
                    min_dist = dist
            
            # 距离奖励（鼓励靠近目标）
            if min_dist < self.arena_size * 0.5:
                for j in range(self.num_pursuers):
                    dist = np.linalg.norm(self.pursuers[j] - self.targets[i])
                    rewards[j] += 0.1 * (1 - dist/(self.arena_size*0.5))  # 标准化距离奖励
        
        # 团队合作奖励
        if captures > 0:
            rewards += captures * 2.0
        
        # 确保奖励为float
        total_reward = float(np.sum(rewards))
        
        # 终止条件
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {'captures': captures}
        
        return self._get_obs(), total_reward, terminated, truncated, info

    # ... (保留原有的渲染和辅助方法)
    def render(self) -> None:
        if self.render_mode is None:
            return
            
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Multi-Agent Pursuit")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
        
        self.screen.fill((255, 255, 255))
        
        # 绘制边界
        border_rect = pygame.Rect(
            0, 0, 
            self.screen_size, self.screen_size
        )
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect, 2)
        
        # 绘制追捕者 (蓝色)
        for pos in self.pursuers:
            pygame.draw.circle(
                self.screen, (0, 0, 255),
                self._scale_position(pos), 
                10
            )
            # 绘制捕获半径
            pygame.draw.circle(
                self.screen, (200, 200, 200),
                self._scale_position(pos), 
                int(self.capture_radius * self.scale), 
                1
            )
        
        # 绘制目标 (红色)
        for pos in self.targets:
            pygame.draw.circle(
                self.screen, (255, 0, 0),
                self._scale_position(pos), 
                8
            )
        
        # 显示步数
        step_text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))
        
        # 显示捕获数
        captures = self._count_current_captures()
        capture_text = self.font.render(f"Captures: {captures}", True, (0, 0, 0))
        self.screen.blit(capture_text, (10, 50))
        
        pygame.display.flip()
        
        if self.render_mode == 'human':
            self.clock.tick(self.metadata['render_fps'])
    
    def _count_current_captures(self) -> int:
        """计算当前时刻的捕获数量"""
        captures = 0
        for i in range(self.num_targets):
            for j in range(self.num_pursuers):
                distance = np.linalg.norm(
                    self.targets[i] - self.pursuers[j]
                )
                if distance < self.capture_radius:
                    captures += 1
                    break  # 每个目标只能被捕获一次
        return captures
    
    def _scale_position(self, pos: np.ndarray) -> Tuple[int, int]:
        # 将坐标从环境空间转换到屏幕空间
        x = (pos[0] + self.arena_size) * self.scale
        y = (self.arena_size - pos[1]) * self.scale
        return (int(x), int(y))
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.font = None

def train_enhanced_model():
    env = EnhancedHuntEnv(render_mode=None)
    
    # 使用MlpPolicy改进网络架构
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # 配置PPO参数
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        tensorboard_log="./hunt_tensorboard/"
    )
    
    # 添加评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 训练模型
    model.learn(
        total_timesteps=500000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # 保存模型
    model.save("enhanced_hunt_model")
    return model

def visualize_model(model_path="enhanced_hunt_model", num_episodes=3):
    env = EnhancedHuntEnv(render_mode='human')
    model = PPO.load(model_path, env=env)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # 训练模型（首次运行时取消注释）
    # model = train_enhanced_model()
    
    # 可视化训练结果
    visualize_model()
