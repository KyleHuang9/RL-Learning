import numpy as np
import pandas as pd

class RL():
    def __init__(
        self,
        action,
        epsilon=0.9,
        alpha=0.1,
        gamma=0.9,
        trace_decay=0.9
    ):
        self.action = action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = trace_decay
        self.q_table = pd.DataFrame(columns=self.action, dtype=float)
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.action),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def choose_action(self, state, mod='train'):
        if mod == 'train':
            self.check_state_exist(state)
            if np.random.uniform(0, 1) > self.epsilon:
                action = np.random.choice(self.action)
            else:
                state_action = self.q_table.loc[state, :]
                action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        return action
    def update(self, *args):
        pass
    def reset(self):
        pass

class QLearning(RL):
    def __init__(
        self,
        action,
        epsilon=0.9,
        alpha=0.1,
        gamma=0.9,
    ):
        super(QLearning, self).__init__(
            action=action, epsilon=epsilon, alpha=alpha, gamma=gamma
        )
    
    def update(self, state, action, R, state_, action_):
        self.check_state_exist(state_)
        q_pred = self.q_table.loc[state, action]
        if state_ != 'Terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, :].max()
        else:
            q_target = R
        self.q_table.loc[state, action] += self.alpha * (q_target - q_pred)
        
class Sarsa(RL):
    def __init__(
        self,
        action,
        epsilon=0.9,
        alpha=0.1,
        gamma=0.9,
    ):
        super(Sarsa, self).__init__(
            action=action, epsilon=epsilon, alpha=alpha, gamma=gamma
        )
    
    def update(self, state, action, R, state_, action_):
        self.check_state_exist(state_)
        q_pred = self.q_table.loc[state, action]
        if state_ != 'Terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, action_]
        else:
            q_target = R
        self.q_table.loc[state, action] += self.alpha * (q_target - q_pred)
        
class SarsaLambda(RL):
    def __init__(self, action, epsilon=0.9, alpha=0.1, gamma=0.9, trace_decay=0.9):
        super().__init__(action, epsilon, alpha, gamma, trace_decay)
        self.trace = self.q_table.copy()
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.action),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.trace = self.trace.append(to_be_append)
            
    def update(self, state, action, R, state_, action_):
        self.check_state_exist(state_)
        q_pred = self.q_table.loc[state, action]
        if state_ != 'Terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, action_]
        else:
            q_target = R
        error = q_target - q_pred
        
        # 记录路径
        # 法一
        # self.trace.loc[state, action] += 1
        # 法二
        self.trace.loc[state, :] *= 0
        self.trace.loc[state, action] = 1
        
        # 更新Q表
        self.q_table += self.alpha * error * self.trace
        
        # 更新路径衰减
        self.trace *= self.gamma * self.lambda_
    
    def reset(self):
        self.trace *= 0
        