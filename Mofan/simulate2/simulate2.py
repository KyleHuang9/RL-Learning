import numpy as np
import time

from maze_env import maze_env
from RL_brain import QLearning

EPSILON = 0.95
GAMMA = 0.9
ALPHA = 0.1

EPOCH = 50
TRAIN_TIME = 0.02
TEST_TIME = 0.5

ACTION = [0, 1, 2, 3]

def train(QLearn=None):
    print("Training begin!!!")
    
    if QLearn == None:
        QLearn = QLearning(ACTION, EPSILON, ALPHA, GAMMA)
    for i in range(EPOCH):
        state = [0, 0]
        step_cnt = 0
        env = maze_env(pos=state.copy())
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
                state = state_.copy()
            # Q表更新
            QLearn.update(state.copy(), action, R, t_state_)
            
            step_cnt += 1
            
            # 显示
            env.show(t=1)
            
            time.sleep(TRAIN_TIME)
        print(f"Epocb {i + 1}: \tStep {step_cnt}, \tState: {game_state}")
    print("\nQLabel:\n", QLearn.q_table)
    
def test(QLearn):
    print("Testing begin!!!")
    
    state = [0, 0]
    step_cnt = 0
    env = maze_env(pos=state.copy())
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
        
        time.sleep(TEST_TIME)
        
if __name__ == "__main__":
    QLearn = QLearning(ACTION, EPSILON, ALPHA, GAMMA)
    train(QLearn)
    time.sleep(2)
    test(QLearn)