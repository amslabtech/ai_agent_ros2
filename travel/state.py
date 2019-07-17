import numpy as np

class State():
    def __init__(self):
        self.keyboard = "None"
        self.speed = 0.5
        self.features = None
        self.start = None
        self.goal = (-6.00, 1.00, 0.00)
        self.pitfalls = [(-7.0,0.0, 0.0), (-7.0,-0.1, 0.0),(2.0, 0.0, 0.0)] + [(x,-2.0,0.0) for x in np.linspace(-8 ,3, 50)] + [(x,2.0,0.0) for x in np.linspace(-8 ,3, 50)]
        self.pos = None
        self.prev_pos = None
        self.ori = None
        self.prev_ori = None
        self.stacked_reward = 0