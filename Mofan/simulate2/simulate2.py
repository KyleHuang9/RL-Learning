import numpy as np
import time
import cv2

from maze_env import maze_env
from RL_brain import QLearning, Sarsa

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1

EPOCH = 50
TRAIN_TIME = 0.0
TEST_TIME = 0.5

ACTION = [0, 1, 2, 3]
INIT_POS = [0, 0]
SIZE = [4, 4]
HELL = [[1,2], [2,1]]
HEAVEN = [[2, 2]]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("progress.mp4", fourcc, 60, (SIZE[0] * 50, SIZE[1] * 50))

def train(env, RL=None):
    print("Training begin!!!")
    
    if RL == None:
        RL = QLearning(ACTION, EPSILON, ALPHA, GAMMA)
    for i in range(EPOCH):
        state = INIT_POS.copy()
        env.reset()
        step_cnt = 0
        is_over = False
        game_state = 0
        
        env.show(t=1)
        time.sleep(TRAIN_TIME)
        
        while not is_over:
            # 选择动作
            action = RL.choose_action(str(state.copy()))
            
            # 环境更新
            state_, R, game_state = env.update(action)
            
            # 判断游戏结束
            if game_state > 0:
                is_over = True
            
            if not is_over:
                t_state_ = str(state_.copy())
            else:
                t_state_ = state_
            
            # 下一动作
            action_ = RL.choose_action(t_state_)
            
            # Q表更新
            RL.update(str(state.copy()), action, R, t_state_, action_)
            
            if not is_over:
                state = state_.copy()
            step_cnt += 1
            
            # 显示
            env.show(t=1)
            
            if step_cnt % 5 == 0 or is_over:
                video.write(env.img)
            time.sleep(TRAIN_TIME)
        print(f"Epocb {i + 1}: \tStep {step_cnt}, \tState: {game_state}")
    print("\nQLabel:\n", RL.q_table)
    
def test(env, RL):
    print("Testing begin!!!")
    
    state = INIT_POS.copy()
    env.reset()
    step_cnt = 0
    is_over = False
    game_state = 0
    
    env.show(t=1)
    time.sleep(TEST_TIME)
    
    while not is_over:
        # 选择动作
        action = RL.choose_action(str(state.copy()), mod='test')
        
        # 环境更新
        state_, R, game_state = env.update(action)
        
        # 判断游戏结束
        if game_state > 0:
            is_over = True
        
        if not is_over:
            state = state_.copy()
        
        step_cnt += 1
        
        # 显示
        env.show(t=1)
        
        for i in range(30):
            video.write(env.img)
        time.sleep(TEST_TIME)
        
if __name__ == "__main__":
    env = maze_env(pos=INIT_POS.copy(), size=SIZE.copy(), hell=HELL.copy(), heaven=HEAVEN.copy())
    RL = Sarsa(ACTION, EPSILON, ALPHA, GAMMA)
    env.reset()
    train(env, RL)
    for i in range(60):
        video.write(np.zeros((SIZE[0] * 50, SIZE[1] * 50), dtype=np.uint8))
    time.sleep(2)
    env.reset()
    test(env, RL)
    RL.q_table.to_csv("q_table.csv")