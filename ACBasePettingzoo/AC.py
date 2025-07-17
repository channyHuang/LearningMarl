from ACEnv import *

class MAACTrainer:
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        
        # 为每个智能体创建独立的网络
        self.policies = {
            agent: ActorCritic(
                input_dim=env.observation_space(agent).shape[0],
                action_dim=env.action_space(agent).n
            ) for agent in self.agents
        }
        
        self.optimizers = {
            agent: optim.Adam(self.policies[agent].parameters(), lr=0.01)
            for agent in self.agents
        }
        
        self.gamma = 0.95  # 折扣因子
        self.eps = np.finfo(np.float32).eps.item()
    
    def select_action(self, agent, obs):
        obs = torch.FloatTensor(obs)
        probs, value = self.policies[agent](obs)
        
        # 创建分类分布并采样动作
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action), value
    
    def update_policy(self, agent, rewards, log_probs, values):
        returns = []
        R = 0
        # 计算折扣回报
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        policy_loss = []
        value_loss = []
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value, torch.tensor([R])))
        
        self.optimizers[agent].zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizers[agent].step()
    
    def train(self, episodes=1000):
        for ep in range(episodes):
            obs = env.reset()
            episode_rewards = {agent: 0 for agent in self.agents}
            
            # 存储每个智能体的经验
            experiences = {agent: {'log_probs': [], 'values': [], 'rewards': []}
                         for agent in self.agents}
            
            while env.agents:
                actions = {}
                for agent in env.agents:
                    action, log_prob, value = self.select_action(agent, obs[agent])
                    actions[agent] = action
                    experiences[agent]['log_probs'].append(log_prob)
                    experiences[agent]['values'].append(value)
                
                obs, rewards, dones, infos = env.step(actions)
                
                for agent in rewards:
                    experiences[agent]['rewards'].append(rewards[agent])
                    episode_rewards[agent] += rewards[agent]
            
            # 更新每个智能体的策略
            for agent in self.agents:
                self.update_policy(
                    agent,
                    experiences[agent]['rewards'],
                    experiences[agent]['log_probs'],
                    experiences[agent]['values']
                )
            
            if ep % 50 == 0:
                print(f"Episode {ep}, Rewards: {episode_rewards}")