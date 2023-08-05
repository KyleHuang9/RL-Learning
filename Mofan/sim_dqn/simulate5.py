import numpy as np
import time
import matplotlib.pyplot as plt

from maze_env import maze_env
from RL_brain import DQN

EPSILON = 0.8
GAMMA = 0.9
LR = 0.01
LAMBDA = 0.9

EPOCH = 1000
START_STEP = 100
LEARN_RATE = 5
TRAIN_TIME = 0.02
TEST_TIME = 0.5

ACTION = [0, 1, 2, 3]
INIT_POS = [0, 0]
SIZE = [4, 4]
HELL = [[1,2], [2,1]]
HEAVEN = [[2, 2]]

def train(env, RL):
    step = 0
    loss_list = []
    for epoch in range(EPOCH):
        # env.reset()
        env.game_state = 0
        total_loss = 0
        loss_cnt = 0
        state = INIT_POS.copy()
        env.show(1)
        time.sleep(TRAIN_TIME)
        
        while True:
            # 获取动作
            action = RL.choose_action(state)
            
            # 获得环境反馈
            state_, reward, game_state = env.feedback(action)
            
            # 储存记忆
            RL.store_transition(state, action, reward, state_)
            
            # 控制学习
            if (step > START_STEP) and (step % LEARN_RATE == 0):
                loss = RL.learn()
                total_loss += loss
                loss_cnt += 1
                loss_list.append(loss)
                
            step += 1
            
            # 更新状态
            state = state_.copy()
            env.update()
            
            # 显示
            env.show(1)
            time.sleep(TRAIN_TIME)
            
            # 判断游戏结束
            if game_state > 0:
                break
            
        print(f"Epoch {epoch + 1}, Loss: {total_loss / loss_cnt if loss_cnt > 0 else 0}")
    return loss_list

def test(env, RL):
    state = INIT_POS.copy()
    env.reset()
    while True:
        action = RL.choose_action(state, mod='test')
        
        state_, reward, game_state = env.feedback(action)
        
        state = state_.copy()
        env.update()
        
        env.show(1)
        time.sleep(TEST_TIME)
        
        if game_state > 0:
            break

if __name__ == "__main__":
    env = maze_env(pos=INIT_POS, size=SIZE, hell=HELL, heaven=HEAVEN)
    RL = DQN(n_actions=len(ACTION), n_features=2, lr=LR, reward_decay=GAMMA, epsilon=EPSILON, memory_size=2000, epsilon_increase=0.001)
    loss_list = train(env, RL)
    env.reset()
    # test(env, RL)
    env.show_arrows(RL)
    
    plt.plot(loss_list)
    plt.show()