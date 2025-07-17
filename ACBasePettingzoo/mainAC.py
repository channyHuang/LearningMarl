import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_tag_v3  # 使用tag环境支持多目标抓捕

# 环境初始化
env = simple_tag_v3.parallel_env(
    num_good=2,      # 两个目标(逃跑者)
    num_adversaries=3,  # 三个抓捕者
    num_obstacles=2,   # 两个障碍物
    max_cycles=100,    # 最大步数
    continuous_actions=False
)

# 定义集中式Critic网络
class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(CentralizedCritic, self).__init__()
        total_actions = sum(action_dims.values())
        
        # 状态处理分支
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 动作处理分支
        self.action_net = nn.Sequential(
            nn.Linear(total_actions, 128),
            nn.ReLU()
        )
        
        # 联合处理
        self.joint_net = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, actions):
        state_out = self.state_net(state)
        action_out = self.action_net(actions)
        combined = torch.cat([state_out, action_out], dim=-1)
        return self.joint_net(combined)

# 定义分散式Actor网络
class DecentralizedActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DecentralizedActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)
    
    def get_action(self, obs):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# 多智能体经验回放缓冲区
class MultiAgentReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """存储经验: (state, obs, actions, rewards, next_state, next_obs, dones)"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

# 多智能体Actor-Critic训练器
class MultiTargetMAACTrainer:
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        
        # 区分抓捕者和目标
        self.predators = [agent for agent in self.agents if 'adversary' in agent]
        self.preys = [agent for agent in self.agents if 'agent' in agent]
        
        print(f"Predators: {self.predators}")
        print(f"Preys: {self.preys}")
        
        # 获取维度信息
        state_dim = env.state_space.shape[0]
        obs_dims = {agent: env.observation_space(agent).shape[0] for agent in self.agents}
        action_dims = {agent: env.action_space(agent).n for agent in self.agents}
        
        # 创建演员网络(每个智能体一个)
        self.actors = {
            agent: DecentralizedActor(obs_dims[agent], action_dims[agent])
            for agent in self.predators  # 只训练抓捕者
        }
        
        # 创建集中评论家网络
        self.critic = CentralizedCritic(state_dim, action_dims)
        
        # 优化器
        self.actor_optimizers = {
            agent: optim.Adam(self.actors[agent].parameters(), lr=0.001)
            for agent in self.predators
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # 超参数
        self.gamma = 0.98  # 折扣因子
        self.tau = 0.01    # 软更新参数
        self.batch_size = 128
        
        # 经验回放缓冲区
        self.buffer = MultiAgentReplayBuffer(capacity=50000)
        
        # 目标网络
        self.target_critic = CentralizedCritic(state_dim, action_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 跟踪训练统计信息
        self.capture_counts = []
    
    def select_actions(self, obs_dict):
        """为每个抓捕者选择动作"""
        actions = {}
        log_probs = {}
        
        for agent in self.predators:
            obs = torch.FloatTensor(obs_dict[agent])
            action, log_prob = self.actors[agent].get_action(obs)
            actions[agent] = action
            log_probs[agent] = log_prob
        
        # 目标使用随机策略
        for agent in self.preys:
            actions[agent] = self.env.action_space(agent).sample()
        
        return actions, log_probs
    
    def update_networks(self):
        """使用经验回放更新网络"""
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样批次
        batch = self.buffer.sample(self.batch_size)
        states, obs_dicts, actions_dicts, rewards_dicts, next_states, next_obs_dicts, dones = batch
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # 准备动作和观察数据
        all_actions = []
        all_next_actions = []
        
        for i in range(self.batch_size):
            # 当前动作
            current_actions = []
            for agent in self.agents:
                current_actions.append(actions_dicts[i][agent])
            all_actions.append(np.concatenate(current_actions))
            
            # 下一动作(使用目标网络计算抓捕者的动作)
            next_agent_actions = {}
            for agent in self.agents:
                if agent in self.predators:
                    next_obs = torch.FloatTensor(next_obs_dicts[i][agent])
                    with torch.no_grad():
                        logits = self.actors[agent](next_obs)
                        probs = F.softmax(logits, dim=-1)
                        next_action = probs.argmax(dim=-1).item()
                    next_agent_actions[agent] = next_action
                else:
                    next_agent_actions[agent] = actions_dicts[i][agent]  # 目标保持原动作
            
            next_actions = []
            for agent in self.agents:
                next_actions.append(next_agent_actions[agent])
            all_next_actions.append(np.concatenate(next_actions))
        
        all_actions = torch.FloatTensor(np.array(all_actions))
        all_next_actions = torch.FloatTensor(np.array(all_next_actions))
        
        # 计算奖励(集中式)
        team_rewards = []
        for rewards in rewards_dicts:
            # 只计算抓捕者的总奖励
            team_reward = sum([rewards[agent] for agent in self.predators])
            team_rewards.append(team_reward)
        team_rewards = torch.FloatTensor(team_rewards).unsqueeze(1)
        
        # 更新评论家
        current_q = self.critic(states, all_actions)
        with torch.no_grad():
            next_q = self.target_critic(next_states, all_next_actions)
            target_q = team_rewards + self.gamma * (1 - dones.unsqueeze(1)) * next_q
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新演员
        actor_losses = {agent: 0 for agent in self.predators}
        
        # 重新选择动作以获取梯度
        new_actions_dict = {}
        new_log_probs = {}
        for agent in self.predators:
            obs_tensor = torch.FloatTensor(np.array([obs_dicts[i][agent] for i in range(self.batch_size)]))
            logits = self.actors[agent](obs_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_actions = dist.sample()
            new_log_probs[agent] = dist.log_prob(new_actions)
            new_actions_dict[agent] = new_actions.detach().cpu().numpy()
        
        # 构建新动作向量
        new_all_actions = []
        for i in range(self.batch_size):
            current_actions = []
            for agent in self.agents:
                if agent in self.predators:
                    current_actions.append(new_actions_dict[agent][i])
                else:
                    current_actions.append(actions_dicts[i][agent])
            new_all_actions.append(np.concatenate(current_actions))
        new_all_actions = torch.FloatTensor(np.array(new_all_actions))
        
        # 计算演员损失
        actor_q = self.critic(states, new_all_actions)
        for agent in self.predators:
            # 最大化Q值减去熵正则化
            actor_loss = -torch.mean(actor_q) - 0.01 * torch.mean(new_log_probs[agent])
            actor_losses[agent] = actor_loss.item()
            
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizers[agent].step()
        
        # 软更新目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_losses
    
    def train(self, episodes=2000):
        for ep in range(episodes):
            obs_dict = env.reset()
            state = env.state()
            episode_rewards = {agent: 0 for agent in self.predators}
            captured_preys = {prey: False for prey in self.preys}
            capture_count = 0
            
            while env.agents:
                # 选择动作
                actions, log_probs = self.select_actions(obs_dict)
                
                # 执行动作
                next_obs_dict, rewards, dones, infos = env.step(actions)
                next_state = env.state()
                
                # 计算抓捕奖励（增强奖励机制）
                for prey in self.preys:
                    if not captured_preys[prey]:
                        # 检查是否被抓捕（距离小于阈值）
                        prey_pos = np.array(next_obs_dict[prey][:2])
                        for predator in self.predators:
                            predator_pos = np.array(next_obs_dict[predator][:2])
                            distance = np.linalg.norm(predator_pos - prey_pos)
                            if distance < 0.1:  # 抓捕阈值
                                captured_preys[prey] = True
                                capture_count += 1
                                # 给予抓捕者额外奖励
                                rewards[predator] += 10.0
                
                # 存储经验
                transition = (
                    state,
                    obs_dict,
                    actions,
                    rewards,
                    next_state,
                    next_obs_dict,
                    all(dones.values())  # 整个回合是否结束
                )
                self.buffer.push(transition)
                
                # 更新状态
                obs_dict = next_obs_dict
                state = next_state
                
                # 累计奖励
                for agent in self.predators:
                    episode_rewards[agent] += rewards[agent]
            
            # 记录抓捕数量
            self.capture_counts.append(capture_count)
            
            # 更新网络
            critic_loss, actor_losses = self.update_networks()
            
            # 打印训练进度
            if ep % 50 == 0:
                avg_reward = sum(episode_rewards.values()) / len(self.predators)
                avg_capture = sum(self.capture_counts[-50:]) / len(self.capture_counts[-50:]) if len(self.capture_counts) >= 50 else 0
                print(f"Episode {ep}: Avg Reward = {avg_reward:.2f}, "
                      f"Critic Loss = {critic_loss:.4f}, "
                      f"Captures = {capture_count}, "
                      f"Avg Captures (last 50) = {avg_capture:.2f}")
        
        print("Training completed!")
    
    def evaluate(self, episodes=10, render=False):
        total_captures = 0
        for ep in range(episodes):
            obs_dict = env.reset()
            captured_preys = {prey: False for prey in self.preys}
            capture_count = 0
            
            while env.agents:
                if render:
                    env.render()
                
                actions = {}
                # 抓捕者使用训练好的策略
                for agent in self.predators:
                    obs = torch.FloatTensor(obs_dict[agent])
                    with torch.no_grad():
                        logits = self.actors[agent](obs)
                        probs = F.softmax(logits, dim=-1)
                        action = probs.argmax(dim=-1).item()
                    actions[agent] = action
                
                # 目标使用随机策略
                for agent in self.preys:
                    actions[agent] = env.action_space(agent).sample()
                
                next_obs_dict, _, dones, _ = env.step(actions)
                
                # 检查抓捕
                for prey in self.preys:
                    if not captured_preys[prey]:
                        prey_pos = np.array(next_obs_dict[prey][:2])
                        for predator in self.predators:
                            predator_pos = np.array(next_obs_dict[predator][:2])
                            distance = np.linalg.norm(predator_pos - prey_pos)
                            if distance < 0.1:
                                captured_preys[prey] = True
                                capture_count += 1
                
                obs_dict = next_obs_dict
                if all(dones.values()):
                    break
            
            total_captures += capture_count
            print(f"Evaluation Episode {ep}: Captures = {capture_count}")
        
        avg_captures = total_captures / episodes
        print(f"Average captures per episode: {avg_captures:.2f}")
        return avg_captures

# 训练和评估
if __name__ == "__main__":
    # 初始化训练器
    trainer = MultiTargetMAACTrainer(env)
    
    # 训练
    print("Starting training...")
    trainer.train(episodes=1000)
    
    # 评估
    print("Evaluating trained policy...")
    avg_captures = trainer.evaluate(episodes=10, render=True)
    
    # 保存模型
    torch.save({
        'actors': {agent: trainer.actors[agent].state_dict() for agent in trainer.predators},
        'critic': trainer.critic.state_dict()
    }, "multi_target_capture_model.pth")
    
    print(f"Final Evaluation: Average captures per episode = {avg_captures:.2f}")
    env.close()