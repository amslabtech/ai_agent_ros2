class Action():

    def __init__(self, agent):
        self.agent = agent
        self.next_state = None

    def act(self):
        pass

class ActionTwist(Action):

    def __init__(self, dict, agent):
        super().__init__(agent)
        self.twist_linear = dict['twist_linear']
        self.twist_angular = dict['twist_angular']

    def act(self):
        speed = self.agent.state.speed
        twist_linear = [x * speed for x in self.twist_linear]
        twist_angular = [x * speed for x in self.twist_angular]
        self.agent._send_twist(twist_linear, twist_angular)

class ActionChangeState(Action):
    def __init__(self, next_state_id, agent):
        super().__init__(agent)
        self.next_state_id = next_state_id
        
    def act(self):
        self.agent.now_state = self.agent.states[self.next_state_id]
        print("Changed to state", self.next_state_id)

class ActionChangeSpeed(Action):
    def __init__(self, val, agent):
        super().__init__(agent)
        self.val = val

    def act(self):
        self.agent.state.speed = max(0, self.agent.state.speed + self.val)
        print("Changed speed to", self.agent.state.speed)