import cv2
import numpy as np
import os
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from maddpg import MADDPG
from Env import Env
from buffer import MultiAgentReplayBuffer

def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

if __name__ == '__main__':
    env = Env()

    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.003, scenario='models',
                           chkpt_dir='./')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = True
    best_score = -np.inf

    if evaluate:
        maddpg_agents.load_checkpoint()
        N_GAMES = 1
        print('----  evaluating  ----')
    else:
        print('----training start----')
    
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False]*n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                env.render()

            actions = maddpg_agents.choose_action(obs,total_steps,evaluate)
            obs_, rewards, dones = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                dones = [True]*n_agents

            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory,total_steps)

            obs = obs_
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i > 0 and avg_score > best_score:
                print('New best score',avg_score ,'>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),'; average target score {:.1f}'.format(avg_target_score))
    