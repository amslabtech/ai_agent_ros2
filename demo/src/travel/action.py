class Action():

    def __init__(self):
        self.next_state = None

class ActionTwist(Action):    

    def __init__(self, dict, traveller):
        super().__init__()
        self.traveller = traveller
        self.twist_linear = dict['twist_linear']
        self.twist_angular = dict['twist_angular']

    def twist(self):
        self.traveller._send_twist(self.twist_linear, self.twist_angular)