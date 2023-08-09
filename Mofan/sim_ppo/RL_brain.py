import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Net(nn.Module):
    def __init__(self, n_features, n_actions, hidden_features=128):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=n_features,
            out_features=hidden_features
        )
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=n_actions
        )
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

class Critic_Net(nn.Module):
    def __init__(self, n_features, hidden_features=128):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=n_features,
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

class PPO(object):
    def __init__(
        self,
        n_features,
        n_action,
        lr_a=0.001,
        lr_c=0.01,
        lmbda=0.95,
        epochs=10,
        eps=0.2,
        gamma=0.9,
    ):
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        
        self.actor = Actor_Net(n_features=n_features, n_actions=n_action)
        self.critic = Critic_Net(n_features=n_features)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.ep_s, self.ep_a, self.ep_r, self.ep_s_, self.ep_d = [], [], [], [], []
    
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action_probs = self.actor(state)
        p = action_probs.squeeze().detach().numpy()
        action = np.random.choice(action_probs.shape[1], p=p)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.ep_s.append(state)
        self.ep_a.append(action)
        self.ep_r.append(reward)
        self.ep_s_.append(state_)
        self.ep_d.append(done)

    def learn(self):
        states = torch.tensor(np.array(self.ep_s), dtype=torch.float)
        actions = torch.tensor(self.ep_a).view(-1, 1)
        rewards = torch.tensor(self.ep_r, dtype=torch.float).view(-1, 1)
        states_ = torch.tensor(np.array(self.ep_s_), dtype=torch.float)
        dones = torch.tensor(self.ep_d, dtype=torch.float).view(-1, 1)
        
        # 数据清空
        self.ep_s, self.ep_a, self.ep_r, self.ep_s_, self.ep_d = [], [], [], [], []
        
        q_next = self.critic(states_)
        td_target = rewards + self.gamma * q_next * (1 - dones)
        td_eval = self.critic(states)
        td_error = td_target - td_eval
        td_error = td_error.detach().numpy()
        
        # 优势函数计算
        running_adv = 0
        advantage = np.zeros_like(td_error)
        for i in reversed(range(len(td_error))):
            running_adv = self.gamma * self.lmbda * running_adv + td_error[i]
            advantage[i] = running_adv
        advantage = torch.tensor(advantage, dtype=torch.float)
        
        # 旧策略概率
        old_probs = self.actor(states).gather(1, actions).detach()
        
        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 新策略概率
            new_probs = self.actor(states).gather(1, actions)
            # 新旧策略概率比值
            ratio = torch.div(new_probs, old_probs)
            # 裁剪的两个比较项
            L_left = ratio * advantage
            L_right = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # 策略网络损失函数
            actor_loss = torch.mean(-torch.min(L_left, L_right))
            # 价值网络损失函数
            td_eval = self.critic(states)
            td_target = rewards + self.gamma * q_next * (1 - dones)
            td_target = td_target.detach()
            critic_loss = torch.mean(torch.square(td_eval - td_target))
            
            # 更新
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()