import gym
from RL_brain import PerDQN

EPOCH = 30
MEMORYSIZE = 10000

def train(env, RL):
    step = 0
    for epoch in range(EPOCH):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            
            if done:
                reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if step > MEMORYSIZE:
                if step == MEMORYSIZE + 1:
                    print("Start Learn!!!")
                RL.learn()

            ep_r += reward
            if done:
                print('\tepoch: ', epoch,
                    '\tep_r: ', round(ep_r, 2),
                    '\tepsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            step += 1

def test(env, RL):
    print("Test begin!!!")
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation, mod='test')
        observation_, reward, done, info = env.step(action)
        if done:
            print("Game over!!!")
            break
        observation = observation_

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(21)
    RL = PerDQN(n_actions=env.action_space.n,
             n_features=env.observation_space.shape[0],
             memory_size=MEMORYSIZE,
             epsilon_increase=0.00005)
    train(env, RL)
    test(env, RL)