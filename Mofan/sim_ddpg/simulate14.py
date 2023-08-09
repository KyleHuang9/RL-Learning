import gym
import numpy as np
import tqdm
from RL_brain import DDPG

EPOCH = 5000
MAX_STEP = 200
RENDER = 500
ENV_NAME = 'Pendulum-v1'

GAMMA = 0.9
SCALE = 2.0
GREEDY = 0.995
LR_A = 0.001
LR_C = 0.01
BATCH_SIZE = 256
MEMORY_SIZE = 100000
REPLACE = [
    dict(name='soft', tau=0.01),
    dict(name='hard', steps=500)
][0]
GREEDY_INCRE = 0.00001

def train(env, RL):
    is_render = False
    for epoch in range(EPOCH):
        total_reward = 0
        state = env.reset()
        for step in tqdm.tqdm(range(MAX_STEP)):
            if is_render:
                env.render()
            action = RL.choose_action(state)
            
            state_, reward, done, info = env.step(action)
            reward -= 5 * (1 + (step) / MAX_STEP)
            
            if abs(state[0] - 1) < 0.05 and abs(state[1]) < 0.05 and abs(state[2]) < 0.1:
                reward += 1 * (1 + (MAX_STEP - step) / MAX_STEP)
            if abs(state[0] - 1) < 0.01 and abs(state[1]) < 0.01 and abs(state[2]) < 0.01:
                reward += 5 * (1 + (MAX_STEP - step) / MAX_STEP)
            if abs(state[0] - 1) < 0.001 and abs(state[1]) < 0.001 and abs(state[2]) < 0.001:
                reward += 20 * (1 + (MAX_STEP - step) / MAX_STEP)
            if state[0] < 0 and abs(state[2]) < 1:
                reward -= 1
            
            RL.store_transition(state, action, reward, state_)
            
            if RL.memory_cnt >= MEMORY_SIZE:
                RL.learn()
            
            state = state_
            total_reward += reward
            step += 1
            
        print(f"Epoch {epoch}: \treward: {round(total_reward, 4)} \tgreedy: {round(RL.greedy, 4)}")
        if total_reward >= RENDER and epoch >= EPOCH * 3 // 4:
            is_render = True
            break
            
def test(env, RL):
    for epoch in range(100):
        total_reward = 0
        state = env.reset()
        for step in tqdm.tqdm(range(MAX_STEP)):
            env.render()
            action = RL.choose_action(state)
            
            state_, reward, done, info = env.step(action)
            
            state = state_
            total_reward += reward
        
        print(f"TEST reward {epoch}: {total_reward / 10}")
            
        
if __name__ == "__main__":
    env = gym.make(ENV_NAME, g=9.81)
    env = env.unwrapped
    env.seed(1)
    
    RL = DDPG(
        n_features=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        scale=env.action_space.high,
        greedy=GREEDY, gamma=GAMMA, lr_a=LR_A, lr_c=LR_C,
        batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE,
        replacement=REPLACE, greedy_per_learn=GREEDY_INCRE
    )
    train(env, RL)
    test(env, RL)
    