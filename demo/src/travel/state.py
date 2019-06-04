import copy

from action import *
from condition import *

class State():

    def __init__(self):
        self.default_conditions = []
        self.default_actions = []
        # self.__init_default_conditions()

    def __init_default_conditions(self):
        condition = Condition()
        self.default_conditions.append(condition)

class StateChild(State):

    def __init__(self):
        super().__init__()
        self.conditions = []
    
    # def get_match_condition_index(self, objects):
    #     for i, cond in enumerate(self.conditions):
    #         if cond.is_match_states(objects):
    #             return i
    #     return -1

    # def check_state(self, img, text):
    #     for i, cond in enumerate(self.conditions):
    #         if cond.is_match_states(objects):
    #             return i
    #     return -1