from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import dill
import random
import rclpy

from agent import Agent, get_unused_dir_num

RES_PATH = 'src/ai_agent_ros2/travel/results'

def softmax(dict):
    vals = np.array(list(dict.values()))
    vals = np.exp(vals)
    val_sum = np.sum(vals)
    vals = vals / val_sum
    for i, key in enumerate(dict.keys()):
            dict[key] = vals[i]
    return dict

class RLAgent(Agent):

    def __init__(self, epsilon=0.10, alpha=0.01, gamma=0.8):
        super().__init__()
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.rl_actions = ["forward", "left", "right"]
        self.done = False

        self.log = []
        self.state.stacked_reward = 0
        self.rand_policy = dict()
        for a in self.rl_actions:
            self.rand_policy[a] = 0
        self.__init_Q()

    def init_dict(self):
        return self.rand_policy

    def __init_Q(self):
        self.Q = defaultdict(self.init_dict)

    def __init_DQN(self):
        pass


    def chooseAction(self, state):
        # epsilon = self.epsilon * (1 / (self.e + 1)) #徐々にランダム行動を減らす
        # # SARSA
        # if random.random() < epsilon:
        #     qv = self.Q[None]
        # else:
        #     qv = self.Q[state]
        # action_prob = softmax(qv) 
        # action_id = np.random.choice(len(self.rl_actions), 1, p=list(action_prob.values()))[0]

        # return self.rl_actions[action_id]

        # Q学習
        if random.random() < self.epsilon:
            action_id = np.random.choice(len(self.rl_actions))
            action_key = self.rl_actions[action_id]
        else:
            qv = self.Q[state]
            action_key = max(qv, key=qv.get)

        return action_key


    def step(self):
        s = self.state.pos
        a = self.chooseAction(s)
        self.actions[a].act()

        reward, self.done = self.calc_reward()

        n_state = self.state.pos
        n_action = self.chooseAction(n_state)  # On-policy
        gain = reward + self.gamma * self.Q[n_state][n_action]
        estimated = self.Q[s][a]
        self.Q[s][a] += self.alpha * (gain - estimated)


    def learn(self):
        STEPS = 1500
        EPISODES = 1000
        # _, w = os.popen('stty size', 'r').read().split()
        w = 60
        for e in range(EPISODES):
            self.e = e
            s = " episode " + str(e) + ' '
            print(s.center(w,'='))
            self.reset()
            for step in range(STEPS):
                if not self.done:
                    rclpy.spin_once(self)
                else:
                    self.log.append(self.state.stacked_reward)
                    break

        WEIGHT_PATH = get_unused_dir_num(RES_PATH)
        os.makedirs(WEIGHT_PATH, exist_ok=True)
        FILE_NAME = "w.pickle"
        with open(os.path.join(WEIGHT_PATH,FILE_NAME), mode='wb') as f:
            dill.dump(dict(self.Q), f)
        print("log:\n",self.log)
        plt.plot(self.log)
        plt.savefig(os.path.join(WEIGHT_PATH,"res.png"))


    def load_weight(self):
        WEIGHT_PATH = os.path.join("003","w.pickle")
        with open(os.path.join(RES_PATH,WEIGHT_PATH), mode='rb') as f:
            self.Q = dill.load(f)
        self.Q = defaultdict(self.init_dict, self.Q)


def main(args=None):
    rclpy.init(args=args)
    agent = RLAgent()
    
    try:
        # agent.load_weight()
        agent.learn()

    except KeyboardInterrupt:
        if agent not in locals():
            agent.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(0)

if __name__ == '__main__':
    main()