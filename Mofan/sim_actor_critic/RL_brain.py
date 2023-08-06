import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Net(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
    
class Critic_Net(nn.Module):
    def __init__(self, in_features, hidden_features=128):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=1
        )
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        output = self.fc2(x)
        return output

class Actor(object):
    def __init__(self, n_features, n_action, lr=0.001):
        self.lr = lr
        self.net = Actor_Net(in_features=n_features, out_features=n_action)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        prob = self.net(state)
        p = prob.squeeze().detach().numpy()
        action = np.random.choice(np.arange(prob.shape[1]), p=p)
        return action
    
    def learn(self, state, action, exp_v):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        prob = self.net(state)
        log_prob = torch.log(prob[0, action])
        loss = (-(log_prob * exp_v)).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
class Critic(object):
    def __init__(self, n_features, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.net = Critic_Net(in_features=n_features)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
    def learn(self, state, reward, state_):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        state_ = torch.unsqueeze(torch.FloatTensor(state_), 0)
        v = self.net(state)
        v_ = self.net(state_)
        exp_v = reward + self.gamma * v_ - v
        loss = torch.square(exp_v)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        with torch.no_grad():
            exp_v = reward + self.gamma * v_ - v
        return exp_v
    
class Actor_Critic(object):
    def __init__(self, n_features, n_action, lr_a=0.001, lr_c=0.01, gamma=0.9):
        self.actor = Actor(n_features=n_features, n_action=n_action, lr=lr_a)
        self.critic = Critic(n_features=n_features, lr=lr_c, gamma=gamma)
    
    def choose_action(self, state):
        return self.actor.choose_action(state)
    
    def learn(self, state, action, reward, state_):
        exp_v = self.critic.learn(state, reward, state_)
        self.actor.learn(state, action, exp_v)