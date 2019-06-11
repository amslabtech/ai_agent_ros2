import os
import cv2
import sys
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
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

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = ("" if pref is None else (
            pref + "_" )) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')

class Agent(Node):

    def __init__(self):
        super().__init__('traveller')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        self.sub_img = self.create_subscription(Image,'/cam/custom_camera/image_raw', self.image_sub)
        self.sub_r_img = self.create_subscription(Image,'/demo/r_image', self.r_image_sub)
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

        self.speed = 0.2


        self.time_period = 0.1
        self.tmr = self.create_timer(self.time_period, self.step)


    def step(self):
        action_probs = defaultdict(int)
        for policy in self.policies:
            action_prob = self.policies[policy].check(self.state)
            for action_name in action_prob:
                action_probs[action_name] = action_prob[action_name]

        if len(action_probs) > 0:
            action_idx = np.random.choice(len(action_probs), 1, p=list(action_probs.values()))[0]
            action_key = list(action_probs.keys())[action_idx]
            self.actions[action_key].act()
        
    def __init_state(self):
        self.state = State()
        pass


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
        policy_keys = ["s_pressed","w_pressed","x_pressed","a_pressed","d_pressed"]
        action_keys = ["stop", "forward", "backward", "left", "right"]
        probs = np.eye(5)
        for i in range(len(keys)):
            action_prob = {}
            for j in range(len(action_keys)):
                action_prob[action_keys[j]] = probs[i][j]
            policy = PolicyKeyboard(keys[i], action_prob)
            self.policies[policy_keys[i]] = policy
        # states = "01"
        # for i in range(len(states)):
        #     policy = PolicyKeyboard(states[i])
        #     self.policies[self.sts[i]] = policy
        # config = "hl"
        # for i in range(len(config)):
        #     policy = PolicyKeyboard(config[i])
        #     self.policies[list(self.speed_controls.keys())[i]] = policy
        

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


    def image_sub(self,oimg):
        # print("image sub")

        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")

        except CvBridgeError as e:
           print(e)

        cv2.imshow("Image windowt",img)
        cv2.waitKey(3)
    
    def objects_sub(self, otext):
        print(otext.data)


    def odom_sub(self, data):
        pos = data.pose.pose.position
        self.position = (
            round(pos.x,2),
            round(pos.y,2),
            round(pos.z,2)
        )
        ori = data.pose.pose.orientation
        self.orientation = (
            round(ori.x,2),
            round(ori.y,2),
            round(ori.z,2),
            round(ori.w,2)
        )
        # self.step()


    def r_image_sub(self,r_img):
        # print("image sub")

        try:
            img = self.bridge.imgmsg_to_cv2(r_img, "bgr8")
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", img)

        except CvBridgeError as e:
           print(e)


    def _send_twist(self, x_linear, z_angular):
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = x_linear
        twist.angular.x, twist.angular.y, twist.angular.z = z_angular
        self.pub.publish(twist)

def main(args=None):

    rclpy.init(args=args)
    agent = Agent()
    try:
        rclpy.spin(agent)
    finally:
        if agent not in locals():
            agent.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
