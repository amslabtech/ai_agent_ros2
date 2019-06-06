import os
import cv2
import sys
import rclpy
import numpy as np
from rclpy.node import Node
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

class Environment():

    def __init__(self):
        pass

class EnvironmentText(Environment):

    def __init__(self):
        super().__init__()
        self.text = ""
        self.text_policy = None

class EnvironmentKeyboard(Environment):
    def __init__(self):
        super().__init__()
        self.keyboard_policy = None

class Agent(Node):

    def __init__(self):
        super().__init__('traveller')
        self.i = 0
        self.j = 0
        self.d = 0
        self.c = 0
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
        # self.sub_obj = self.create_subscription(String,'/objects', self.object_detection_sub)

        self.policies = []
        self.actions = []
        self.states = []
        self.objects = []
        self.environments = {}

        self.__init_environment()
        self.__init_action()
        self.__init_policy()
        self.__init_state()

        self.now_state = self.states[0]

    def __init_environment(self):
        self.environments['text'] = EnvironmentText()
        self.environments['keyboard'] = EnvironmentKeyboard()

    def __init_state(self):
        State.default_actions.append(self.actions[0])
        for j in range(8):
            policy = self.policies[j]
            policy.set_action(self.actions[j+1])
            State.default_policies.append((policy))
        state = StateChild()
        self.states.append(state)
        state = StateChild()
        state.args['speed'] = 0.4
        self.states.append(state)


    def __init_policy(self):
        moves = "wsad"
        for i in range(4):
            policy = PolicyKeyboard(moves[i])
            self.policies.append(policy)
        policy = PolicyKeyboard('x')
        self.policies.append(policy)
        policy = PolicyKeyboard('None')
        self.policies.append(policy)
        states = "01"
        for i in range(len(states)):
            policy = PolicyKeyboard(states[i])
            self.policies.append(policy)


    def __init_action(self):

        action = Action(self)
        self.actions.append(action)

        xs = [1.0, -1.0, 0.0, 0.0]
        zs = [0.0, 0.0, 1.0, -1.0]

        for i in range(len(xs)):
            action = ActionTwist({
                    "twist_linear" : (xs[i], 0.0, 0.0),
                    "twist_angular" : (0.0, 0.0, zs[i])
                }, self)
            self.actions.append(action)

        action = ActionTwist({
                "twist_linear" : (0.0, 0.0, 0.0),
                "twist_angular" : (0.0, 0.0, 0.0)
            }, self)
        self.actions.append(action)
        self.actions.append(action)
        for i in range(2):
            action = ActionChangeState(i, self)
            self.actions.append(action)



    def command(self):
        action = None
        for policy in self.now_state.policies + State.default_policies:
            if policy.check(self.environments):
                action = policy.action
                break

        if action is None:
            action = self.now_state.default_actions[0]

        action.act()

    def text_sub(self, otext):
        self.environments['text'].text_policy = int(otext.data)
        print("env text_policy",self.environments['text'].text_policy)
        self.command()

    def keyboard_sub(self, key):
        key_str = key.data
        if key_str[0] == "'" or key_str[0] == '"':
            key_str = key_str[1:-1]
        elif key_str != 'None':
            key_str = key_str[4:]
        self.environments['keyboard'].keyboard_policy = key_str
        print("env keyboard_policy",self.environments['keyboard'].keyboard_policy)
        self.command()


    def image_sub(self,oimg):
        print("image sub")

        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")

        except CvBridgeError as e:
           print(e)

        cv2.imshow("Image windowt",img)
        cv2.waitKey(3)
    
    def objects_sub(self, otext):
        print(otext.data)

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
