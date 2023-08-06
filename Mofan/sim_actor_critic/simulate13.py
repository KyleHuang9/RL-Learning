import gym
import numpy as np
import torch
import torch.nn as nn
from RL_brain import Actor_Critic

EPOCH = 3000
MAX_STEPS = 10000
RENDER_THRESHOLD = 8000
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.001

def train(env, RL):
    for i in range(EPOCH):
        step = 0
        render = False
        total_reward = 0
        state = env.reset()
        while True:
            if render:
                env.render()
            
            action = RL.choose_action(state)
            
            state_, reward, done, info = env.step(action)
            
            if done:
                reward = -20
                
            total_reward += reward
            
            RL.learn(state, action, reward, state_)
            
            state = state_
            step += 1
            if done or step >= MAX_STEPS:
                if total_reward >= RENDER_THRESHOLD:
                    render = True
                print(f"Epoch {i}, reward: {total_reward}")
                break

def test(env, RL):
    print("Test begin!!!")
    step = 0
    total_reward = 0
    state = env.reset()
    while True:
        env.render()
        
        action = RL.choose_action(state)
        
        state_, reward, done, info = env.step(action)
        
        if done:
            reward = -20
            
        total_reward += reward
        
        state = state_
        step += 1
        if done or step >= 2 * MAX_STEPS:
            print(f"TEST Reward: {total_reward}")
            break

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env.seed(1)
    env = env.unwrapped
    RL = Actor_Critic(
        n_features=env.observation_space.shape[0],
        n_action=env.action_space.n,
        lr_a=LR_A, lr_c=LR_C, gamma=GAMMA
    )
    train(env, RL)
    test(env, RL)