import numpy as np
import pandas as pd

class QLearning():
    def __init__(
        self,
        action,
        epsilon=0.9,
        alpha=0.1,
        gamma=0.9,
    ):
        self.action = action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
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
        state = str(state)
        
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
    
    def update(self, state, action, R, state_):
        state, state_ = str(state), str(state_)
        self.check_state_exist(state_)
        q_pred = self.q_table.loc[state, action]
        if state_ != 'Terminal':
            q_target = R + self.gamma * self.q_table.loc[state_, :].max()
        else:
            q_target = R
        self.q_table.loc[state, action] += self.alpha * (q_target - q_pred)