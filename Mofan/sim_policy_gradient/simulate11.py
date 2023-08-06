import gym
from RL_brain import Policy_gradient
import matplotlib.pyplot as plt

EPOCH = 5000
RENDER_THRESHOLD = 10000

def train(env, RL):
    render = False
    for epoch in range(EPOCH):
        observation = env.reset()
        ep_r = 0
        while True:
            if render:
                env.render()
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward)

            ep_r += reward
            if ep_r > RENDER_THRESHOLD:
                render = True
                break
            if done:
                print('\tepoch: ', epoch,
                    '\tep_r: ', round(ep_r, 2))
                vt = RL.learn()
                if epoch == 0:
                    plt.plot(vt)    # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            observation = observation_
        if render:
            break

def test(env, RL):
    print("Test begin!!!")
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if done:
            print("Game over!!!")
            break
        observation = observation_
        

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = env.unwrapped
    RL = Policy_gradient(n_actions=env.action_space.n,
             n_features=env.observation_space.shape[0],
             lr=0.001, reward_decay=0.9)
    train(env, RL)
    test(env, RL)