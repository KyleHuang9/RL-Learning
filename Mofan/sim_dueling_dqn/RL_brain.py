import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_feature, output_feature, base_feature=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=base_feature),
            nn.ReLU(),
            nn.Linear(in_features=base_feature, out_features=2 * base_feature),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(in_features=2 * base_feature, out_features=base_feature),
            nn.ReLU(),
            nn.Linear(in_features=base_feature, out_features=1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features=2 * base_feature, out_features=base_feature),
            nn.ReLU(),
            nn.Linear(in_features=base_feature, out_features=output_feature),
        )
        
    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean()

class DuelingDQN():
    def __init__(
        self,
        n_actions,
        n_features,
        lr=0.01,
        reward_decay=0.9,
        epsilon=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        epsilon_increase=None,
    ):
        # 初始化学习参数
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increase = epsilon_increase
        self.epsilon = 0 if self.epsilon_increase != None else self.epsilon_max
        
        # 初始化网络
        self.eval_net = Net(input_feature=n_features, output_feature=n_actions)
        self.target_net = Net(input_feature=n_features, output_feature=n_actions)
        
        # 用于记录是否需要替换target net的参数
        self.learn_step_cnt = 0
        
        # 初始化记忆
        self.memory = np.zeros([self.memory_size, n_features * 2 + 2])
        
        # 复制eval_net参数
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # 初始化迭代器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        
        # 初始化损失函数
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, state, mod="train"):
        state = torch.unsqueeze(torch.FloatTensor(state.copy()), 0) # [1, state_size]
        
        if mod == "train":
            if np.random.uniform() < self.epsilon:
                action_values = self.eval_net(state).detach().numpy()
                action = np.argmax(action_values)
            else:
                action = np.random.randint(0, self.n_actions)
        else:
            action_values = self.eval_net(state).detach().numpy()
            action = np.argmax(action_values)
        
        return action
    
    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_cnt'):
            self.memory_cnt = 0
        
        transition = np.hstack((state, [action, reward], state_))
        
        index = self.memory_cnt % self.memory_size
        self.memory[index, :] = transition
        
        self.memory_cnt += 1
        
    def learn(self):
        # 替换target net参数
        if self.learn_step_cnt % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_cnt += 1
        
        # 抽取batch_size
        if self.memory_cnt > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_cnt, size=self.batch_size)
        batch_memory = self.memory[batch_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_action = torch.Tensor(batch_memory[:, self.n_features : self.n_features + 1]).type(dtype=torch.int64)
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_features + 1 : self.n_features + 2])
        batch_state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])
        
        # 计算q值
        q_eval = self.eval_net(batch_state)
        q_eval = q_eval.gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        
        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        
        # 参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = self.epsilon + self.epsilon_increase if self.epsilon + self.epsilon_increase < self.epsilon_max else self.epsilon_max
        
        return loss.detach().item()