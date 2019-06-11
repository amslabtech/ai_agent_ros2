import random

class Policy():
    # self.key_action_prob = {}
    def __init__(self):
        pass

    def set_action(self, action):
        self.action = action

    def check(self, state):
        return False

class PolicyText(Policy):
    
    def __init__(self, dict):
        super().__init__()
        self.index = dict['index']

    def check(self, state):
        return self.index == state['text'].text_Policy
        
class PolicyKeyboard(Policy):
    def __init__(self, key_str, act_prob):
        super().__init__()
        self.key_str = key_str
        self.act_prob = act_prob
    
    def check(self, state):
        if(state.keyboard == self.key_str):
            return self.act_prob
        else:
            return {}