import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self, input_feature, output_feature, hidden_feature=512):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=input_feature,
            out_features=hidden_feature // 2
        )
        self.fc2 = nn.Linear(
            in_features=hidden_feature // 2,
            out_features=hidden_feature
        )
        self.fc3 = nn.Linear(
            in_features=hidden_feature,
            out_features=hidden_feature // 2,
        )
        self.fc4 = nn.Linear(
            in_features=hidden_feature // 2,
            out_features=output_feature
        )
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        output = self.fc4(x)
        return output
    
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, weights):
        return (weights * torch.pow((pred - target), 2)).sum()

class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    
    # 添加数据
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    # 更新树
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    # 获得v值下数据
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
            
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):
    def __init__(self, capacity):
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.abs_err_upper = 1.0
        self.tree = SumTree(capacity)
    
    # 存数据   
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)
    
    # 取数据
    def sample(self, batch_size):
        b_idx, b_memory, weights = \
            np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, self.tree.data[0].size)), np.empty((batch_size, 1))
        pri_seg = self.tree.total_p / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 求max(P(j))
        max_prob = np.max(np.power(self.tree.tree[-self.tree.capacity:] / self.tree.total_p, -self.beta))
        for i in range(batch_size):
            # 区间大小
            a, b = pri_seg * i, pri_seg * (i + 1)
            # 从区间中随机取值
            v = np.random.uniform(a, b)
            # 取得v下对应数据
            idx, p, data = self.tree.get_leaf(v)
            # 求概率
            prob = p / self.tree.total_p
            # 求w，N可以约掉
            weights[i, 0] = np.power(prob, -self.beta) / max_prob
            # 存储结果
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, weights
    
    # 更新树
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon # 防止除零
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for t, p in zip(tree_idx, ps):
            self.tree.update(t, p)

class PerDQN():
    def __init__(
        self,
        n_actions,
        n_features,
        lr=0.005,
        reward_decay=0.9,
        epsilon=0.9,
        replace_target_iter=500,
        memory_size=10000,
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
        self.memory = Memory(memory_size)
        
        # 复制eval_net参数
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # 初始化迭代器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        
        # 初始化损失函数
        self.loss_func = MSELoss()
        
    def choose_action(self, state, mod="train"):
        state = torch.unsqueeze(torch.FloatTensor(state.copy()), 0) # [1, state_size]
        
        if np.random.uniform() < self.epsilon:
            action_values = self.eval_net(state).detach().numpy()
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)
        
        return action
    
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, [action, reward], state_))
        self.memory.store(transition)
        
    def learn(self):
        # 替换target net参数
        if self.learn_step_cnt % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("Target model update!!!")
        self.learn_step_cnt += 1
        
        # 抽取batch_size
        tree_idx, batch_memory, weights = self.memory.sample(self.batch_size)
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_action = torch.Tensor(batch_memory[:, self.n_features : self.n_features + 1]).type(dtype=torch.int64)
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_features + 1 : self.n_features + 2])
        batch_state_ = torch.FloatTensor(batch_memory[:, -self.n_features:])
        
        # 计算q值
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach()
        
        # 预测下一状态动作Q值
        q_eval2next = self.eval_net(batch_state_).detach()
        # 选下一状态最大价值动作
        pred_action = torch.argmax(q_eval2next, dim=1, keepdim=True)
        q_next_choose = q_next.gather(1, pred_action)
        q_target = batch_reward + self.gamma * q_next_choose
        
        # 计算损失
        weights = torch.from_numpy(weights)
        loss = self.loss_func(q_eval, q_target, weights)
        
        # 计算TD-errors
        abs_TD_errors = abs(q_eval - q_target).detach().numpy()
        
        # 更新记忆
        self.memory.batch_update(tree_idx, abs_TD_errors)
        
        # 参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = self.epsilon + self.epsilon_increase if self.epsilon + self.epsilon_increase < self.epsilon_max else self.epsilon_max
        
        return loss.detach().item()