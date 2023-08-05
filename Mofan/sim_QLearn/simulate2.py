import numpy as np
import time
import cv2

from maze_env import maze_env
from RL_brain import QLearning

EPSILON = 0.999
GAMMA = 0.99
ALPHA = 0.1

EPOCH = 300
TRAIN_TIME = 0.0
TEST_TIME = 0.5

ACTION = [0, 1, 2, 3]
INIT_POS = [0, 0]
SIZE = [10, 10]
HELL = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 6], [2, 7], [2, 8],
        [3, 3], [3, 4], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
        [5, 2], [5, 6], [6, 2], [6, 4], [6, 6], [6, 8], [7, 1], [7, 2], [7, 4],
        [7, 6], [7, 8], [8, 4], [8, 6], [8, 8], [9, 4], [9, 8]]
HEAVEN = [[9, 9]]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("progress.mp4", fourcc, 60, (SIZE[0] * 50, SIZE[1] * 50))

def train(env, QLearn=None):
    print("Training begin!!!")
    
    if QLearn == None:
        QLearn = QLearning(ACTION, EPSILON, ALPHA, GAMMA)
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
            action = QLearn.choose_action(state.copy())
            
            # 环境更新
            state_, R, game_state = env.update(action)
            
            # 判断游戏结束
            if game_state > 0:
                is_over = True
            
            if not is_over:
                t_state_ = state_.copy()
            else:
                t_state_ = state_
            # Q表更新
            QLearn.update(state.copy(), action, R, t_state_)
            
            if not is_over:
                state = state_.copy()
            step_cnt += 1
            
            # 显示
            env.show(t=1)
            
            if step_cnt % 5 == 0 or is_over:
                video.write(env.img)
            time.sleep(TRAIN_TIME)
        print(f"Epocb {i + 1}: \tStep {step_cnt}, \tState: {game_state}")
    print("\nQLabel:\n", QLearn.q_table)
    
def test(env, QLearn):
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
        action = QLearn.choose_action(state.copy(), mod='test')
        
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
    QLearn = QLearning(ACTION, EPSILON, ALPHA, GAMMA)
    env.reset()
    train(env, QLearn)
    for i in range(60):
        video.write(np.zeros((SIZE[0] * 50, SIZE[1] * 50), dtype=np.uint8))
    time.sleep(2)
    env.reset()
    test(env, QLearn)
    QLearn.q_table.to_csv("q_table.csv")