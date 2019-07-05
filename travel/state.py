class State():
    def __init__(self):
        self.keyboard = "None"
        self.speed = 0.2
        self.features = None
        self.start = None
        self.goal = (-6.81, -8.21, 0.52)
        self.pos = None
        self.ori = None
        self.prev_pos = None
        self.prev_ori = None
        self.stacked_reward = 100