import gymnasium
import numpy as np
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from BaseEnv import BaseEnv

def train():
    env = BaseEnv(render_mode = None)
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
    env = BaseEnv(render_mode = 'human')
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
