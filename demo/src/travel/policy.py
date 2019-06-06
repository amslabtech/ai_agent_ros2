class Policy():

    def __init__(self):
        pass

    def set_action(self, action):
        self.action = action

    def check(self):
        return False
    

class PolicyText(Policy):
    
    def __init__(self, dict):
        super().__init__()
        self.index = dict['index']

    def check(self, env):
        return self.index == env['text'].text_Policy
        
class PolicyKeyboard(Policy):
    def __init__(self, key_str):
        super().__init__()
        self.key_str = key_str
    
    def check(self, env):
        return self.key_str == env['keyboard'].keyboard_policy

'''
class PolicyImage(Policy):
    
    def __init__(self, dict):
        super().__init__()
        self.should_exist_objects = []
        self.should_not_exist_objects = []

    def is_match_states(self, objects):
        should_exist_objects_matched = [False] * len(self.should_exist_objects)
        should_not_exist_objects_matched = False

        for obj in objects:
            for i, cond_object in enumerate(self.should_exist_objects):
                should_exist_objects_matched[i] |= self.is_match(cond_object, obj)
        
        for obj in objects:
            for i, cond_object in enumerate(self.should_not_exist_objects):
                should_not_exist_objects_matched |= self.is_match(cond_object, obj)

        return sum(should_exist_objects_matched) == len(self.should_exist_objects) and not should_not_exist_objects_matched

    def is_match(self, cond_object, obj):
        print("Policy:",cond_object) 
        print("Detected Object",obj)
        if(cond_object["class_name"] != obj["class_name"]):
            return False
        if(cond_object["area_lower_bound"] > obj["area"] or cond_object["area_upper_bound"] < obj["area"]):
            return False
        if(cond_object["center_x_lower_bound"] > obj["center_x"] or cond_object["center_x_upper_bound"] < obj["center_x"]):
            return False
        if(cond_object["center_y_lower_bound"] > obj["center_y"] or cond_object["center_y_upper_bound"] < obj["center_y"]):
            return False
        if(cond_object["score_lower_bound"] > obj["score"] or cond_object["score_upper_bound"] < obj["score"]):
            return False
        return True

    def check(self, env):
        self.is_match_states
'''