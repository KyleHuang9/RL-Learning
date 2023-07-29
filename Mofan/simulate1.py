import numpy as np
import time

WORLD_SIZE = 20
ACTION = [0, 1] # [left, right]
EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
EPOCH = 20
TRAIN_TIME=0.01
TEST_TIME=0.5

# 生成q表
def build_q_table():
    q_table = np.zeros((WORLD_SIZE - 1, len(ACTION)), dtype=float)
    return q_table

# 获得行为
def get_action(state, q_table, epsilon):
    # 当前状态q表行
    q_state = q_table[state, :]
    if np.random.uniform(0, 1) > epsilon:
        action = np.random.choice(ACTION)
    else:
        # 选取最大价值
        choice = np.where(q_state==np.max(q_state))[0]
        action = ACTION[np.random.choice(choice)]
    return action

# 获得收益，只有在到达目标后有收益
def feedback(state, action):
    if action == 1:
        if state == WORLD_SIZE - 2:
            _state = 'Terminal'
            R = 1
        else:
            _state = state + 1
            R = 0
    else:
        # 撞墙
        if state == 0:
            _state = 0
            R = 0
        else:
            _state = state - 1
            R = 0
    return _state, R

# 更新环境
def update_env(state, epoch, step_cnt, time_sleep):
    env_list = ['-'] * (WORLD_SIZE - 1) + ['T']
    if state == 'Terminal':
        output = f"Epoch {epoch}: Reach Target! Step: {step_cnt}"
        print('\r{}'.format(output), end='')
        time.sleep(1)
        print('\r' + ' ' * 100, end='')
    else:
        env_list[state] = 'O'
        output = ''.join(env_list)
        print('\r{}'.format(output), end='')
        time.sleep(time_sleep)

# 训练
def train():
    epsilon = EPSILON
    q_table = build_q_table()
    for i in range(EPOCH):
        state = 0
        step_cnt = 0
        is_terminated = False
        update_env(state, i, step_cnt, TRAIN_TIME)
        while not is_terminated:
            action = get_action(state, q_table, epsilon)
            state_, R = feedback(state, action)
            # 更新q表
            q_pred = q_table[state, action]
            if state_ != 'Terminal':
                q_target = R + GAMMA * np.max(q_table[state_, :])
            else:
                q_target = R
                is_terminated = True
            q_table[state, action] += ALPHA * (q_target - q_pred)
            
            # 状态更新
            state = state_
            
            # 环境更新
            step_cnt += 1
            update_env(state, i, step_cnt, TRAIN_TIME)
            
    return q_table

# 测试
def test(q_table):
    epsilon = 1.0
    state = 0
    step_cnt = 0
    is_terminated = False
    update_env(state, 0, step_cnt, TEST_TIME)
    while not is_terminated:
        action = get_action(state, q_table, epsilon)
        state, R = feedback(state, action)
        if state == 'Terminal':
            is_terminated = True
        step_cnt += 1
        update_env(state, 0, step_cnt, TEST_TIME)

if __name__ == '__main__':
    print('\r{}'.format("Train begin!!!"), end='')
    time.sleep(2)
    print('\r' + ' ' * max(WORLD_SIZE, len("Train begin!!!")), end='')
    q_table = train()
    time.sleep(2)
    print('\r{}'.format("Test begin!!!"), end='')
    time.sleep(2)
    print('\r' + ' ' * max(WORLD_SIZE, len("Test begin!!!")), end='')
    test(q_table)
    print("\nq_table:\n", q_table)
    