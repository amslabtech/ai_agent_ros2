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
from condition import *
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
        self.text_condition = None

class EnvironmentKeyboard(Environment):
    def __init__(self):
        super().__init__()
        self.keyboard_condition = None

class Agent(Node):

    def __init__(self):
        super().__init__('traveller')
        self.i = 0
        self.j = 0
        self.d = 0
        self.c = 0
        self.bridge = CvBridge()
        # self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        # self.sub_img = self.create_subscription(Image,'/cam/custom_camera/image_raw', self.image_sub)
        # turtlebot3
        self.pub = self.create_publisher(Twist, '/cmd_vel')
        self.sub_img = self.create_subscription(Image,'/tb3/camera/image_raw', self.image_sub)
        self.sub_txt = self.create_subscription(String,'/condition_text', self.text_sub)
        self.sub_key = self.create_subscription(String,'/keyboard', self.keyboard_sub)
        data_folder = "/home/swatanabe/src/sq_control_robot_tracking_ros2/ai_travel/src/travel/src/yolo/"
        
        # yolo_args = {
        #     "model_path": data_folder+"yolo.h5",
        #     "score": 0.01,
        #     "anchors_path": data_folder+"anchors.txt",
        #     "classes_path": data_folder+"classes.txt",
        #         }
        # self.yolo_args = dict((k, v) for k, v in yolo_args.items() if v)
        # self.model = YOLO(**self.yolo_args)

        self.conditions = []
        self.actions = []
        self.states = []
        self.objects = []
        self.environments = {}

        self.__init_environment()
        self.__init_action()
        self.__init_condition()
        self.__init_state()

        self.now_state = self.states[0]

    def __init_environment(self):
        self.environments['text'] = EnvironmentText()
        self.environments['keyboard'] = EnvironmentKeyboard()

    def __init_action(self):
        xs = [0.2, -0.2, 0.0, 0.0]
        zs = [0.0, 0.0, 0.2, -0.2]

        for i in range(len(xs)):
            action = ActionTwist({
                    "twist_linear" : (xs[i], 0.0,0.0),
                    "twist_angular" : (0.0, 0.0, zs[i])
                }, self)
            self.actions.append(action)

        action = ActionTwist({
                "twist_linear" : (0.0, 0.0,0.0),
                "twist_angular" : (0.0, 0.0, 0.0)
            }, self)
        self.actions.append(action)

    def __init_condition(self):
        for i in range(4):
            condition = ConditionText({
                    "index" : i
                })
            self.conditions.append(condition)

        moves = "wsad"
        for i in range(4):
            condition = ConditionKeyboard(moves[i])
            self.conditions.append(condition)
        condition = ConditionKeyboard('None')
        self.conditions.append(condition)

    def __init_state(self):
        for i in range(5):
            state = StateChild()
            state.default_actions.append(self.actions[i])
            for j in range(8):
                condition = self.conditions[j]
                condition.set_action(self.actions[j % 4])
                state.conditions.append((condition))
            condition = self.conditions[8]
            condition.set_action(self.actions[4])
            state.conditions.append((condition))
            self.states.append(state)

        for i in range(5):
            self.actions[i].next_state = self.states[i]

    def command(self):
        action = None
        for condition in self.now_state.conditions:
            if condition.check(self.environments):
                action = condition.action
                self.now_state = action.next_state
                break

        if action is None:
            action = self.now_state.default_actions[0]

        action.twist()

    def text_sub(self, otext):
        self.environments['text'].text_condition = int(otext.data)
        print("env text_condition",self.environments['text'].text_condition)
        self.command()

    def keyboard_sub(self, key):
        key_str = key.data
        if key_str[0] == "'" or key_str[0] == '"':
            key_str = key_str[1:-1]
        elif key_str != 'None':
            key_str = key_str[4:]
        self.environments['keyboard'].keyboard_condition = key_str
        print("env keyboard_condition",self.environments['keyboard'].keyboard_condition)
        self.command()


    def image_sub(self,oimg):
        print("image sub")

        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
            hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('%d.jpg' % self.j, img)

        except CvBridgeError as e:
           print(e)

        #cv2.imshow("Image windowt",img)
        cv2.imshow("Image windowt1", hsv_img)
        cv2.waitKey(3)

    '''
    def locate(self, oimg):
        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
            hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Object detection
            image=PIL_Image.fromarray(img)
            """
            out_boxes, out_scores, out_classes, _ = self.model.detect_raw(image)

            annotation = []
            for box, score, class_ in zip(out_boxes, out_scores, out_classes):
                predicted_class = self.model.class_names[class_]
                box = box.tolist()
                annotation.append({
                    "bbox": [box[1], box[0], box[3], box[2]],
                    "score": float(score),
                    "class": predicted_class,
                    "class_id": int(class_),
                })
            font = ImageFont.load_default()

            thickness = (image.size[0] + image.size[1]) // 300
            
            self.objects = []

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.model.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                center_y = (top + bottom) / 2.0
                center_x = (left + right) / 2.0
                area = (bottom - top) * (right - left)

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
    
                top, left, bottom, right = box
    
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.model.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.model.colors[c])
                draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
                del draw

                print({'area' : area , 'center_x' : center_x, 'center_y' : center_y, "class_name" : predicted_class, "score" : score})
                self.objects.append({'area' : area , 'center_x' : center_x, 'center_y' : center_y, "class_name" : predicted_class, "score" : score})
            
            """
            self.i += 1

            self.command()

            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

        except CvBridgeError as e:
           print(e)

        #cv2.imshow("Image windowt",img)
        cv2.imshow("Image windowt1", hsv_img)
        cv2.waitKey(3)
    '''

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
