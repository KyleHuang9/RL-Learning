import numpy as np
import matplotlib.pyplot as plt
import tqdm

k = 10
epoch = 1000
items = 2000
phi1 = 0.1

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

def phi_greedy_one_item(phi):
    total_value = np.zeros(shape=k, dtype=float)
    mean_value = np.zeros(shape=k, dtype=float)
    choose_time = np.zeros(shape=k, dtype=int)
    final_value = 0

    # 贪心算法
    for i in range(epoch):
        
        # 正常情况下贪心
        if np.random.uniform(0, 1) > phi:
            # 选取最大收益
            choice = np.where(mean_value==np.max(mean_value))[0]

            # 最大收益不止一个，随机选取一个
            if choice.size != 1:
                choice = choice[np.random.randint(0, choice.size)]
        # 以phi的概率非贪心
        else:
            choice = np.random.randint(0, k)
        
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

def train():
    final_values = 0
    phi_final_values = 0
    choice_cnt = [0 for _ in range(k)]
    phi_choice_cnt = [0 for _ in range(k)]

    plt_final_values = []
    plt_final_choice = []
    plt_phi_final_values = []
    plt_phi_final_choice = []

    tq = tqdm.tqdm(range(items))

    for i in tq:
        final_value, choice = greedy_one_item()
        phi_final_value, phi_choice = phi_greedy_one_item(phi=phi1)

        final_values += final_value
        choice_cnt[choice] += 1
        phi_final_values += phi_final_value
        phi_choice_cnt[phi_choice] += 1

        plt_final_values.append(final_values / (i + 1))
        plt_final_choice.append(choice_cnt[true_choice] / (i + 1))
        plt_phi_final_values.append(phi_final_values / (i + 1))
        plt_phi_final_choice.append(phi_choice_cnt[true_choice] / (i + 1))
    final_values /= items
    phi_final_values /= items
    final_choice = np.argmax(choice_cnt)
    phi_final_choice = np.argmax(phi_choice_cnt)

    print("\ntrue:")
    print("value: ", true_values)
    print("choice: ", true_choice)
    print("\ngreedy:")
    print("value: ", final_values)
    print("choice: ", final_choice)
    print("\nphi_greedy:")
    print("value: ", phi_final_values)
    print("choice: ", phi_final_choice)

    x = [_ + 1 for _ in range(items)]
    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("values")
    plt.plot(x, plt_final_values, x, plt_phi_final_values)
    plt.show()

    plt.figure(2)
    plt.xlabel("epoch")
    plt.ylabel("values")
    plt.plot(x, plt_final_choice, x, plt_phi_final_choice)
    plt.show()

if __name__ == "__main__":
    train()