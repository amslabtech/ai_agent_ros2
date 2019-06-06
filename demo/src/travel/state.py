import copy

from action import *
from policy import *

class State():
    default_policies = []
    default_actions = []

    def __init__(self):

        # self.__init_default_policies()
        pass

    def __init_default_policies(self):
        policy = Policy()
        State.default_policies.append(policy)

class StateChild(State):

    def __init__(self):
        super().__init__()
        self.args = {'speed': 0.2}
        self.policies = []
    
    # def get_match_policy_index(self, objects):
    #     for i, cond in enumerate(self.policies):
    #         if cond.is_match_states(objects):
    #             return i
    #     return -1

    # def check_state(self, img, text):
    #     for i, cond in enumerate(self.policies):
    #         if cond.is_match_states(objects):
    #             return i
    #     return -1