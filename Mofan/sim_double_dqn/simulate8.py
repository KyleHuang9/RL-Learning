import gym
import numpy as np
import tqdm
from RL_brain import DoubleDQN

EPOCH = 10
ITEMS = 5000
ACTION_SPACE = 31

def train(env, RL):
    step = 0
    for epoch in range(EPOCH):
        observation = env.reset()
        for _ in tqdm.tqdm(range(ITEMS)):
            env.render()
            action = RL.choose_action(observation)
            
            f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)

            observation_, reward, done, info = env.step(np.array([f_action]))

            RL.store_transition(observation, action, reward, observation_)

            if step > 1000:
                if step == 1001:
                    print("Start Learning!!!")
                RL.learn()

            observation = observation_
            step += 1

def test(env, RL):
    print("Test begin!!!")
    observation = env.reset()
    for step in tqdm.tqdm(range(ITEMS)):
        env.render()
        action = RL.choose_action(observation, mod='test')
        
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)

        observation_, reward, done, info = env.step(np.array([f_action]))
        observation = observation_
        

if __name__ == "__main__":
    env = gym.make('Pendulum-v1', g=9.81)
    env = env.unwrapped
    RL = DoubleDQN(n_actions=ACTION_SPACE,
             n_features=env.observation_space.shape[0],
             lr=0.01, reward_decay=0.9, epsilon=0.9, batch_size=64,
             replace_target_iter=100, memory_size=2000, epsilon_increase=0.0005)
    train(env, RL)
    test(env, RL)