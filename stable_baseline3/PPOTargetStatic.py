import gymnasium
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import time
from typing import Any, Dict, List, Optional, Tuple

class PPOTargetStaticEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, num_pursuers = 3, num_targets = 2, render_mode = None):
        super().__init__()

        self.num_pursuers = num_pursuers
        self.num_targets = num_targets

        self.arena_size = 10.0
        self.capture_radius = 0.5
        self.max_steps = 200
        self.render_mode = render_mode

        self.action_space = gymnasium.spaces.Box(
            low = -1.0, high = 1.0, shape=(self.num_pursuers * 2, ), dtype = np.float32)

        self.observation_space = gymnasium.spaces.Box(
            low = -np.inf, high = np.inf, 
            shape=(self.num_pursuers * self.num_targets * 2 + self.num_pursuers * 2,),
            dtype = np.float32
        )

        self.pursuers = np.zeros((self.num_pursuers, 2), dtype = np.float32)
        self.targets = np.zeros((self.num_targets, 2), dtype = np.float32)
        self.current_step = 0

        self.screen = None
        self.clock = None
        self.screen_size = 600
        self.scale = self.screen_size / (self.arena_size * 2)

        self.captures = np.zeros(self.num_targets)

        self.reset()

    def reset(self, seed = (int)(time.time())):
        super().reset(seed = seed)
        self.current_step = 0
        self.captures = np.zeros(self.num_targets)

        self.pursuers = np.random.uniform(-self.arena_size, self.arena_size, (self.num_pursuers, 2))
        self.targets = np.random.uniform(-self.arena_size, self.arena_size, (self.num_targets, 2))

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for pursuer in self.pursuers:
            for target in self.targets:
                obs.extend(target - pursuer)
        obs.extend(self.pursuers.flatten())
        return np.array(obs, dtype = np.float32)

    def step(self, actions):
        self.current_step += 1

        actions = actions.reshape((self.num_pursuers, 2))

        self.pursuers += actions * 0.25

        rewards = np.zeros(self.num_pursuers)
        captures = 0

        for i in range(self.num_targets):
            if self.captures[i] != 0:
                continue
            for j in range(self.num_pursuers):
                dist = np.linalg.norm(self.pursuers[j] - self.targets[i])
                if dist < self.capture_radius:
                    self.captures[i] = 1
                    rewards[j] += 10.0
                    captures += 1

                if dist < self.arena_size * 0.5:
                    rewards[j] += 0.1 * (1 - dist / (self.arena_size * 0.5))

        total_reward = float(np.sum(rewards))

        terminated = all(self.captures != 0)
        truncated = self.current_step >= self.max_steps
        info = {'captures': captures}

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self, bSave = False) -> None:
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('HG title')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        self.screen.fill((255, 255, 255))

        # 
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
        
        step_text = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

        if bSave:
            pygame.image.save(self.screen, f"frames/{self.current_step:04d}.png")

        pygame.display.flip()

        if self.render_mode == 'human':
            self.clock.tick(self.metadata['render_fps'])

    def _scale_position(self, pos: np.ndarray) -> Tuple[int, int]:
        x = (pos[0] + self.arena_size) * self.scale
        y = (self.arena_size - pos[1]) * self.scale
        return (int(x), int(y))

    def close(self)->None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.font = None

def train():
    env = PPOTargetStaticEnv(render_mode = None)
    policy_kwargs = dict(net_arch = dict(pi = [256, 256], vf = [256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs = policy_kwargs,
        verbose = 1,
        n_steps = 2048,
        batch_size = 64, 
        gamma = 0.99,
        gae_lambda = 0.95,
        ent_coef = 0.01,
        learning_rate = 3e-4,
        clip_range = 0.2,
        max_grad_norm = 0.5,
        tensorboard_log = "./logs/"
    )

    eval_callback = EvalCallback(env, 
        best_model_save_path = "./models/",
        log_path = './logs/',
        eval_freq = 10000,
        deterministic = True,
        render = False
    )

    model.learn(total_timesteps = 500000,
        callback = eval_callback,
        progress_bar = True
    )

    model.save('models/PPOTargetStatic')
    return model

def predict(model_path = 'models/PPOTargetStatic'):
    env = PPOTargetStaticEnv(render_mode = 'human')
    model = PPO.load(model_path, env = env)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic = True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
    env.close()

if __name__ == '__main__':
    model = train()
    predict()
