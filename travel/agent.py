import json
import os
import cv2
import sys
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from yolo3 import YOLO
from PIL import Image as PIL_Image
from PIL import ImageDraw
from PIL import ImageFont

from action import *
from policy import *
from state import *

from collections import defaultdict

#cascPath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
#faceCascade = cv2.CascadeClassifier(cascPath)
#video_capture = cv2.VideoCapture(1)

def reset_simulation(reset_sim,agent):
    while not reset_sim.wait_for_service(timeout_sec=1.0):
        agent.get_logger().info('/reset_simulation service not available, waiting again...')

    reset_future = reset_sim.call_async(Empty.Request())
    rclpy.spin_until_future_complete(agent, reset_future)

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = ("" if pref is None else (
            pref + "_" )) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')

def dict_normalize(dict):
    val_sum = sum(list(dict.values()))
    if val_sum == 0:
        pass
    else:
        for key in dict.keys():
            dict[key] /= val_sum
    return dict

class Agent(Node):

    def __init__(self):
        super().__init__('traveller')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        self.sub_img_list = []
        self.sub_detected_img_list = []

        detected_image_names = ['/demo/custom_camera/detected_image']
        camera_names = ['/cam/custom_camera/image_raw']

        for name in camera_names:
            self.sub_img_list.append(self.create_subscription(Image, name, self.image_sub_closure(name)))

        for name in detected_image_names:
            self.sub_detected_img_list.append(self.create_subscription(Image, name, self.r_image_sub_closure(name)))
        # turtlebot3
        # self.pub = self.create_publisher(Twist, '/cmd_vel')
        # self.sub_img = self.create_subscription(Image,'/tb3/camera/image_raw', self.image_sub)
        self.sub_txt = self.create_subscription(String,'/demo/policy_text', self.text_sub)
        self.sub_key = self.create_subscription(String,'/demo/keyboard', self.keyboard_sub)
        self.sub_objects = self.create_subscription(String,'/demo/objects', self.objects_sub)
        self.sub_odom = self.create_subscription(Odometry,'/pos/odom_pos', self.odom_sub)

        self.policies = dict()
        self.actions = dict()
        # self.states = []
        self.objects = []
        self.states = {}

        self.__init_state()
        self.__init_action()
        self.__init_policy()
        self.__init_state()

        self.time_period = 0.1
        self.tmr = self.create_timer(self.time_period, self.step)


    def __init_state(self):
        self.state = State()


    def __init_action(self):

        self.move_keys = ["stop", "forward", "backward", "left", "right"]
        xs = [0.0, 1.0, -1.0, 0.0, 0.0]
        zs = [0.0, 0.0, 0.0, 1.0, -1.0]

        for i in range(len(self.move_keys)):
            action = ActionTwist({
                    "twist_linear" : (xs[i], 0.0, 0.0),
                    "twist_angular" : (0.0, 0.0, zs[i])
                }, self)
            self.actions[self.move_keys[i]] = action

        self.sts = ["st0", "st1"]
        for i in range(len(self.sts)):
            action = ActionChangeState(i, self)
            self.actions[self.sts[i]] = action

        self.speed_controls = {"up":0.2, "down":-0.2}
        for key in self.speed_controls.keys():
            action = ActionChangeSpeed(self.speed_controls[key],self)
            self.actions[key] = action


    def __init_policy(self):
        keys = "swxad"
        action_keys = ["stop", "forward", "backward", "left", "right"]
        policy_keys = ["s_pressed","w_pressed","x_pressed","a_pressed","d_pressed"]
        probs = np.eye(5)
        for i in range(len(keys)):
            action_prob = {}
            for j in range(len(action_keys)):
                action_prob[action_keys[j]] = probs[i][j]
            policy = PolicyKeyboard(keys[i], action_prob)
            self.policies[policy_keys[i]] = policy

        keys = "r"
        action_keys = action_keys[1:]
        policy_keys = ["r_pressed"]
        probs = np.ones(len(action_keys)) / len(action_keys)
        for i in range(len(keys)):
            action_prob = {}
            for j in range(len(action_keys)):
                action_prob[action_keys[j]] = probs[j]
            policy = PolicyKeyboard(keys[i], action_prob)
            self.policies[policy_keys[i]] = policy

        keys = "hl"
        action_keys = ["up", "down"]
        policy_keys = ["h_pressed","l_pressed"]
        probs = np.eye(2)
        for i in range(len(keys)):
            action_prob = {}
            for j in range(len(action_keys)):
                action_prob[action_keys[j]] = probs[i][j]
            policy = PolicyKeyboard(keys[i], action_prob)
            self.policies[policy_keys[i]] = policy

        policy = PolicyKeyboard("None", {"stop": 1.0})
        self.policies["keyboard_released"] = policy

    
    def reset(self):
        self.state.stacked_reward = 100
        print("reset")


    def step(self):
        # action_key = get_action_key()
        action_key = self.random_action()
        self.actions[action_key].act()

        reward = self.calc_reward()
        print("reward", reward)
        print("stacked reward",self.state.stacked_reward)
        self.state.stacked_reward += reward


    def random_action(self):
        action_probs = self.policies["r_pressed"].act_prob
        action_idx = np.random.choice(len(action_probs), 1, p=list(action_probs.values()))[0]
        action_key = list(action_probs.keys())[action_idx]

        return action_key


    def get_action_key(self):
        action_probs = defaultdict(int)
        for policy in self.policies:
            action_prob = self.policies[policy].check(self.state)
            for action_name in action_prob:
                action_probs[action_name] += action_prob[action_name]
        action_probs = dict_normalize(action_probs)
        # print("action probs", action_probs)  

        if len(action_probs) > 0:
            action_idx = np.random.choice(len(action_probs), 1, p=list(action_probs.values()))[0]
            action_key = list(action_probs.keys())[action_idx]
        
        return action_key

    def text_sub(self, otext):
        self.states['text'].text_policy = int(otext.data)
        print("state text_policy",self.states['text'].text_policy)
        # self.step()


    def keyboard_sub(self, key):
        key_str = key.data
        if key_str[0] == "'" or key_str[0] == '"':
            key_str = key_str[1:-1]
        elif key_str != 'None':
            key_str = key_str[4:]
        print(key_str)
        self.state.keyboard = key_str


    def image_sub_closure(self, name):

        def image_sub(oimg):
            print("image sub")
            try:
                img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
            except CvBridgeError as e:
                print(e)
            cv2.imshow(name, img)
            cv2.waitKey(3)
        return image_sub
    

    def objects_sub(self, otext):
        res = json.loads(otext.data)
        print(res)
        objects = res['objects']
        if 'feature' in res.keys():
            features = res['feature']
            self.state.features = np.array(features, dtype = 'float')
        print(objects)


    def odom_sub(self, data):
        pos = data.pose.pose.position
        self.state.prev_pos = self.state.pos
        self.state.pos = (
            round(pos.x,3),
            round(pos.y,3),
            round(pos.z,3)
        )
        ori = data.pose.pose.orientation
        self.state.prev_ori = self.state.ori
        self.state.ori = (
            round(ori.x,3),
            round(ori.y,3),
            round(ori.z,3),
            round(ori.w,3)
        )
        # print(self.state.prev_pos,"→",self.state.pos)


    def r_image_sub_closure(self, name):
        def r_image_sub(r_img):
            print("r_image_sub_closure")
            try:
                img = self.bridge.imgmsg_to_cv2(r_img, "bgr8")
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.imshow(name, img)
            except CvBridgeError as e:
                print(e)
        return r_image_sub


    def _send_twist(self, x_linear, z_angular):
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = x_linear
        twist.angular.x, twist.angular.y, twist.angular.z = z_angular
        self.pub.publish(twist)
    

    def calc_reward(self, eps=0.05):
        def calc_dist(p1, p2):
            diff = None
            if p1 is not None and p2 is not None:
                diff = 0.0
                for i in range(3):
                    diff += (p1[i] - p2[i])**2
                diff = np.sqrt(diff)
            return diff

        # 
        reward = -1 # for living

        diff = calc_dist(self.state.prev_pos, self.state.pos)
        if diff is not None and diff < eps:
            print("Stucked!")
            reward = -10 # for stack

        diff = calc_dist(self.state.pos, self.state.goal)
        if diff is not None and diff < 3.0:
            print("Goal!")
            reward = 10 # for goal

        return reward



def main(args=None):

    rclpy.init(args=args)
    agent = Agent() 
    reset_sim = agent.create_client(Empty, '/reset_simulation')
    
    STEPS = 10000
    EPISODES = 100

    try:
        for e in range(EPISODES):
            for step in range(STEPS):
                if agent.state.stacked_reward > 0:
                    rclpy.spin_once(agent)
                else:
                    agent.reset()
                    reset_simulation(reset_sim, agent)
                    break

    except KeyboardInterrupt:
        if agent not in locals():
            agent.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
