import gym
import numpy as np
from RL_brain import PPO

EPOCH = 3000
MAX_STEP = 10000
RENDER = 10000

GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01
LAMBDA = 0.95
EPOCH_PER_TRAIN = 20
EPS = 0.2

def train(env, RL):
    render = False
    max_step_cnt = 0
    for epoch in range(EPOCH):
        step = 0
        total_reward = 0
        state = env.reset()
        while True:
            if render:
                env.render()
            
            action = RL.choose_action(state)
            
            state_, reward, done, info = env.step(action)
            
            reward = -(abs(state[0]) ** 2) - (abs(state[1]) / 10) - (abs(state[2]) * 10) ** 3 - (abs(state[3]) / 10)
            reward += 1
            if abs(state[0]) < 0.2 and abs(state[1]) < 0.2 and abs(state[2]) < 0.1 and abs(state[3]) < 0.2:
                reward += 1
            if abs(state[0]) < 0.1 and abs(state[1]) < 0.1 and abs(state[2]) < 0.04 and abs(state[3]) < 0.1:
                reward += 10
            if abs(state[0]) < 0.01 and abs(state[1]) < 0.01 and abs(state[2]) < 0.01 and abs(state[3]) < 0.01:
                reward += 20
            if done:
                reward -= 500
            
            RL.store_transition(state, action, reward, state_, done)
            
            state = state_
            total_reward += reward
            step += 1
            
            if done or step >= MAX_STEP:
                RL.learn()
                if step >= MAX_STEP:
                    max_step_cnt += 1
                else:
                    max_step_cnt = 0
                print(f"Epoch {epoch}: \treward: {round(total_reward)}")
                # if total_reward >= RENDER:
                #     render = True
                break
        if max_step_cnt >= 100:
            break

def test(env, RL):
    for epoch in range(100):
        step = 0
        total_reward = 0
        state = env.reset()
        while True:
            env.render()
            
            action = RL.choose_action(state)
            
            state_, reward, done, info = env.step(action)
            
            state = state_
            total_reward += reward
            
            if done or step >= 0.1 * MAX_STEP:
                print(f"TEST: \treward: {round(total_reward)}")
                break

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    env.seed(1)
    
    RL = PPO(
        n_features=env.observation_space.shape[0],
        n_action=env.action_space.n,
        lr_a=LR_A, lr_c=LR_C, lmbda=LAMBDA,
        epochs=EPOCH_PER_TRAIN, eps=EPS, gamma=GAMMA
    )
    
    train(env, RL)
    test(env, RL)
    