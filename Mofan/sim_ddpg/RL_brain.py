import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Net(nn.Module):
    def __init__(self, in_features, action_dim, scale, hidden_features=128):
        super().__init__()
        self.scale = scale
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=action_dim
        )
        self.act = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        output = torch.from_numpy(self.scale) * self.tanh(x)
        return output
    
class Critic_Net(nn.Module):
    def __init__(self, in_features, action_dim, hidden_features=128):
        super().__init__()
        self.action = nn.Linear(
            in_features=action_dim,
            out_features=hidden_features
        )
        self.state = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.output = nn.Linear(
            in_features=hidden_features,
            out_features=1
        )
        self.act = nn.ReLU()
        
    def forward(self, state, action):
        action = self.action(action)
        state = self.state(state)
        output = self.output(self.act(action + state))
        return output

class DDPG(object):
    def __init__(
        self,
        n_features,
        action_dim,
        scale,
        greedy=0.9,
        gamma=0.9,
        lr_a=0.001,
        lr_c=0.01,
        batch_size=32,
        memory_size=10000,
        replacement=dict(name='soft', tau=0.01),
        greedy_per_learn=None,
    ):
        # 初始化参数
        self.n_features = n_features
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.scale = scale
        self.max_greedy = greedy
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replacement = replacement
        self.greedy_per_learn = greedy_per_learn
        self.greedy = 0 if self.greedy_per_learn is not None else self.max_greedy
        
        # 初始化网络
        self.actor = Actor_Net(in_features=n_features, action_dim=action_dim, scale=scale)
        self.actor_target = Actor_Net(in_features=n_features, action_dim=action_dim, scale=scale)
        self.critic = Critic_Net(in_features=n_features, action_dim=action_dim)
        self.critic_target = Critic_Net(in_features=n_features, action_dim=action_dim)
        
        # 初始化迭代器
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        # 初始化记忆储存器
        self.memory = np.zeros([self.memory_size, n_features * 2 + action_dim + 1])
        self.memory_pointer = 0
        self.memory_cnt = 0
        
        # hard更新模型步数
        self.replace_cnt = 0
    
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action = self.actor(state).detach().squeeze().numpy() \
            + np.random.uniform(-(1 - self.greedy) * self.scale, (1 - self.greedy) * self.scale)
        return action
    
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action, [reward], state_))
        self.memory[self.memory_pointer, :] = transition
        
        self.memory_pointer += 1
        self.memory_cnt += 1
        self.memory_pointer = self.memory_pointer % self.memory_size
        self.memory_cnt = self.memory_size if self.memory_cnt >= self.memory_size else self.memory_cnt
        
    def sample(self):
        index = np.random.choice(self.memory_cnt, size=self.batch_size)
        return self.memory[index, :]
    
    def learn(self):
        # 更新target net的参数
        # 缓慢更新
        if self.replacement['name'] == 'soft':
            tau = self.replacement['tau']
            actor_layers = self.actor_target.named_children()
            critic_layers = self.critic_target.named_children()
            for name, at_layer in actor_layers:
                if hasattr(at_layer, 'weight'):
                    at_layer.weight.data.mul_((1 - tau))
                    at_layer.weight.data.add_(tau * self.actor.state_dict()[name + '.weight'])
                if hasattr(at_layer, 'bias'):
                    at_layer.bias.data.mul_((1 - tau))
                    at_layer.bias.data.add_(tau * self.actor.state_dict()[name + '.bias'])
            for name, ct_layer in critic_layers:
                if hasattr(ct_layer, 'weight'):
                    ct_layer.weight.data.mul_((1 - tau))
                    ct_layer.weight.data.add_(tau * self.critic.state_dict()[name + '.weight'])
                if hasattr(ct_layer, 'bias'):
                    ct_layer.bias.data.mul_((1 - tau))
                    ct_layer.bias.data.add_(tau * self.critic.state_dict()[name + '.bias'])
        # 隔步复制    
        else:
            if self.replace_cnt % self.replacement['steps'] == 0:
                self.replace_cnt = 0
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic_target.load_state_dict(self.critic.state_dict())
            self.replace_cnt += 1
        
        # 抽取记忆
        batch_memory = self.sample()
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_action = torch.FloatTensor(batch_memory[:, self.n_features : self.n_features + self.action_dim])
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_features + self.action_dim : -self.n_features])
        batch_state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])
        
        # 训练actor
        action = self.actor(batch_state)
        value = self.critic(batch_state, action)
        actor_loss = -torch.mean(value)
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        
        # 训练critic
        action_target = self.actor_target(batch_state_)
        value_ = self.critic_target(batch_state_, action_target)
        value_target = batch_reward + self.gamma * value_
        value_eval = self.critic(batch_state, batch_action)
        critic_loss = torch.square(value_target - value_eval).mean()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        
        # 更新greedy
        self.greedy = self.greedy + self.greedy_per_learn \
            if self.greedy + self.greedy_per_learn < self.max_greedy \
            else self.max_greedy