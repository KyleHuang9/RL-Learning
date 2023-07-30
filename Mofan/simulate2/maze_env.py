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
        self.pos = pos
        self.update()
    
    def draw_background(self):
        background = np.zeros((self.size[0] * self.scale, self.size[1] * self.scale, 3))
        
        # 画格线
        for i in range(self.size[0]):
            cv2.line(background, (i * self.scale, 0), (i * self.scale, self.size[1] * self.scale),
                     (255, 255, 255), 1)
        for i in range(self.size[1]):
            cv2.line(background, (0, i * self.scale), (self.size[0] * self.scale, i * self.scale),
                     (255, 255, 255), 1)
            
        # 画地狱
        for hell in self.hell:
            cv2.rectangle(background, (hell[0] * (self.scale + 1), hell[1] * (self.scale + 1)),
                          ((hell[0] + 1) * (self.scale - 1), (hell[1] + 1) * (self.scale - 1)), (0, 0, 255), -1)
        
        # 画天堂
        for heaven in self.heaven:
            cv2.circle(background, (int((heaven[0] + 0.5) * self.scale), int((heaven[1] + 0.5) * self.scale)),
                       int(self.scale / 2 - 2), (0, 255, 255), -1)
        return background

    def update(self, action=None):
        # up
        if action == 0:
            self.pos[1] -= 1
        # down
        elif action == 1:
            self.pos[1] += 1
        # left
        elif action == 2:
            self.pos[0] -= 1
        # right
        elif action == 3:
            self.pos[0] += 1
        
        # 判断撞墙
        self.pos[0] = np.array(self.pos[0]).clip(0, self.size[0] - 1).item()
        self.pos[1] = np.array(self.pos[1]).clip(0, self.size[1] - 1).item()
        
        # 判断胜负
        if self.pos in self.hell:
            self.game_state = 1
        if self.pos in self.heaven:
            self.game_state = 2
            
        # 计算奖惩
        if self.game_state == 0:
            reward = 0
            state_ = self.pos
        elif self.game_state == 1:
            reward = -1
            state_ = 'Terminal'
        else:
            reward = 1
            state_ = 'Terminal'
        
        # 画出所在位置
        self.img = self.background.copy()
        self.img = cv2.circle(self.img, (int((self.pos[0] + 0.5) * self.scale), int((self.pos[1] + 0.5) * self.scale)),
                       int(self.scale / 4), (255, 255, 255), -1)
        
        return state_, reward, self.game_state
    
    def show(self, t):
        cv2.imshow("maze", self.img)
        cv2.waitKey(t)