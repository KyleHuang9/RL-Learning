import cv2
import numpy as np

class maze_env():
    def __init__(self, pos=[0, 0], size=[4, 4], hell=[[1,2], [2,1]], heaven=[[2, 2]], scale=50):
        self.size = size
        self.hell = hell
        self.heaven = heaven
        self.scale = scale
        self.action = [0, 1, 2, 3] # [up, down, left, right]
        self.game_state = 0  # 0: 正在游戏; 1:地狱 2:天堂
        self.background = self.draw_background()
        self.img = self.background
        self.init_pos = pos.copy()
        self.pos = pos.copy()
        self.next_pos = pos.copy()
        self.update()
    
    def draw_background(self):
        background = np.zeros((self.size[0] * self.scale, self.size[1] * self.scale, 3), dtype=np.uint8)
        
        # 画格线
        for i in range(self.size[0]):
            cv2.line(background, (i * self.scale, 0), (i * self.scale, self.size[1] * self.scale),
                     (255, 255, 255), 1)
        for i in range(self.size[1]):
            cv2.line(background, (0, i * self.scale), (self.size[0] * self.scale, i * self.scale),
                     (255, 255, 255), 1)
            
        # 画地狱
        for hell in self.hell:
            cv2.rectangle(background, (hell[0] * self.scale + 1, hell[1] * self.scale + 1),
                          ((hell[0] + 1) * self.scale - 1, (hell[1] + 1) * self.scale - 1), (0, 0, 255), -1)
        
        # 画天堂
        for heaven in self.heaven:
            cv2.circle(background, (int((heaven[0] + 0.5) * self.scale), int((heaven[1] + 0.5) * self.scale)),
                       int(self.scale / 2 - 2), (0, 255, 255), -1)
        return background

    def feedback(self, action=None):
        # up
        if action == 0:
            self.next_pos[1] -= 1
        # down
        elif action == 1:
            self.next_pos[1] += 1
        # left
        elif action == 2:
            self.next_pos[0] -= 1
        # right
        elif action == 3:
            self.next_pos[0] += 1
        
        # 判断胜负
        if self.next_pos in self.hell:
            self.game_state = 1
        if self.next_pos in self.heaven:
            self.game_state = 2
        # if self.next_pos[0] >= self.size[0] or self.next_pos[1] >= self.size[1] or self.next_pos[0] <= 0 or self.next_pos[1] <= 0:
        #     self.game_state = 1
        
        # 撞墙不动
        self.next_pos[0] = np.array(self.next_pos[0]).clip(0, self.size[0] - 1).item()
        self.next_pos[1] = np.array(self.next_pos[1]).clip(0, self.size[1] - 1).item()
            
        # 计算奖惩
        if self.game_state == 0:
            reward = 0
        elif self.game_state == 1:
            reward = -1
            self.next_pos = self.pos.copy()
        else:
            reward = 1
            self.next_pos = self.init_pos.copy()
        
        # 下一位置
        state_ = self.next_pos.copy()
        
        return state_, reward, self.game_state

    # 环境更新
    def update(self):
        # 更新位置
        self.pos = self.next_pos.copy()
        # 画出所在位置
        self.img = self.background.copy()
        self.img = cv2.circle(self.img, (int((self.pos[0] + 0.5) * self.scale), int((self.pos[1] + 0.5) * self.scale)),
                       int(self.scale / 4), (255, 255, 255), -1)
    
    def show(self, t=1):
        cv2.imshow("maze", self.img)
        cv2.waitKey(t)
    
    def reset(self):
        self.pos = self.init_pos.copy()
        self.next_pos = self.init_pos.copy()
        self.game_state = 0
        self.update()

    def show_arrows(self, RL):
        cols = self.size[0]
        rows = self.size[1]
        self.img = self.background.copy()
        for i in range(cols):
            for j in range(rows):
                state = [i, j]
                action = RL.choose_action(state, mod='test')
                if action == 0:
                    state_ = [i, j - 0.5]
                elif action == 1:
                    state_ = [i, j + 0.5]
                elif action == 2:
                    state_ = [i - 0.5, j]
                elif action == 3:
                    state_ = [i + 0.5, j]
                cv2.line(self.img, (int((i + 0.5) * self.scale), int((j + 0.5) * self.scale)),
                                    (int((state_[0] + 0.5) * self.scale), int((state_[1] + 0.5) * self.scale)),
                                    (0, 0, 255), 2)
                #cv2.circle(self.img, (int((state_[0] + 0.5) * self.scale), int((state_[1] + 0.5) * self.scale)), 2, (0, 0, 255), -1)
        self.show(t=0)