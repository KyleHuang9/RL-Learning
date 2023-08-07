import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

class Net(nn.Module):
    def __init__(self, input_feature, output_feature, hidden_feature=256):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=input_feature,
            out_features=hidden_feature
        )
        self.fc2 = nn.Linear(
            in_features=hidden_feature,
            out_features=output_feature,
        )
        self.dropout = nn.Dropout(0.5)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.dropout(self.fc1(x)))
        output = F.softmax(self.fc2(x), dim=1)
        return output

class Policy_gradient():
    def __init__(
        self,
        n_actions,
        n_features,
        lr=0.01,
        reward_decay=0.95,
    ):
        # 初始化学习参数
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.policy_net = Net(input_feature=n_features, output_feature=n_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 初始化记忆
        self.ep_obs, self.ep_a, self.ep_r = [], [], []
        
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state.copy()), 0) # [1, state_size]
        probs = self.policy_net(state)
        p = probs.squeeze().detach().numpy()
        action = np.random.choice(np.arange(probs.shape[1]), p=p)
        return action
    
    def store_transition(self, state, action, reward):
        self.ep_obs.append(state)
        self.ep_a.append(action)
        self.ep_r.append(reward)
    
    def learn(self):
        discounted_ep_r = np.zeros_like(self.ep_r)
        
        # 累加
        running_add = 0
        for i in reversed(range(len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[i]
            discounted_ep_r[i] = running_add
        
        # 标准化
        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)
        discounted_ep_r = torch.from_numpy(discounted_ep_r)
        
        # 更新
        self.optimizer.zero_grad()
        probs = self.policy_net(torch.tensor(np.array(self.ep_obs)))
        action_prob = probs.gather(1, torch.tensor(self.ep_a).unsqueeze(1)).squeeze()
        loss = -(torch.log(action_prob) * discounted_ep_r).mean()
        loss.backward()
        self.optimizer.step()
        
        self.ep_obs, self.ep_a, self.ep_r = [], [], []
        return discounted_ep_r.numpy()