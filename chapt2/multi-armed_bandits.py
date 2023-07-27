import numpy as np
import matplotlib.pyplot as plt
import tqdm

k = 10
epoch = 1000
items = 2000
phi = 0.1

true_values = np.random.normal(loc=0, scale=1, size=k)
true_choice = np.argmax(true_values)

# 获得实际收益
def get_values():
    values = []
    for i in range(k):
        t = np.random.normal(loc=true_values[i], scale=1, size=1).item()
        values.append(t)
    values = np.array(values)
    return values

def greedy_one_item():
    total_value = np.zeros(shape=k, dtype=float)
    mean_value = np.zeros(shape=k, dtype=float)
    choose_time = np.zeros(shape=k, dtype=int)
    final_value = 0

    # 贪心算法
    for i in range(epoch):
        # 选取最大收益
        choice = np.where(mean_value==np.max(mean_value))[0]

        # 最大收益不止一个，随机选取一个
        if choice.size != 1:
            choice = choice[np.random.randint(0, choice.size)]
        
        # 获取实际收益
        values = get_values()
        value = values[choice]
        choose_time[choice] += 1
        total_value[choice] += value
        final_value += true_values[choice]

        # 计算平均价值估计
        for j in range(k):
            mean_value[j] = total_value[j] / choose_time[j] if choose_time[j] != 0 else 0
    
    final_value /= epoch
    
    return final_value, np.argmax(choose_time)

def greedy_train():
    final_values = 0
    choice_cnt = [0 for _ in range(k)]
    tq = tqdm.tqdm(range(items))
    for i in tq:
        final_value, choice = greedy_one_item()
        final_values += final_value
        choice_cnt[choice] += 1
    final_values /= items
    print("final: ", final_values)
    print("choice: ", choice_cnt)

print("true_val: ", true_values)
print("true_choi: ", true_choice)
greedy_train()