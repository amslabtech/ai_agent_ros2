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
from object_detection.yolo import YOLO
from PIL import Image as PIL_Image
from PIL import ImageDraw
from PIL import ImageFont
import colorsys
import matplotlib.pyplot as plt



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

class Action():
    
    def __init__(self):
        self.twists = []

class DemoYolo(Node):

    def __init__(self):
        super().__init__('traveller')
        self.i = 0
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        self.sub = self.create_subscription(Image,'/cam/custom_camera/image_raw', self.locate)
        data_folder = "src/travel/object_detection/model_data/yolo3/coco/"
        
        yolo_args = {
            "model_path": data_folder+"yolo.h5",
            "score": 0.01,
            "anchors_path": data_folder+"anchors.txt",
            "classes_path": data_folder+"classes.txt",
                }

        self.yolo_args = dict((k, v) for k, v in yolo_args.items() if v)
        self.model = YOLO(**self.yolo_args)
        
        class_num = 0
        classes_path = os.path.expanduser(self.yolo_args["classes_path"])
        with open(classes_path) as f:
            class_num = len(f.readlines())
        hsv_tuples = [(x / class_num, 1., 1.)
                        for x in range(class_num)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        self.colors = colors

    def locate(self,oimg):
        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
            hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Object detection
            image = PIL_Image.fromarray(img)

            result = self.model.detect_image(image)

            objects = result['objects']

            font = ImageFont.load_default()
            thickness = (image.size[0] + image.size[1]) // 300

            for obj in reversed(objects):

                top, left, bottom, right = obj['bbox']
                score = obj['score']
                class_name = obj['class_name']
                classs_id = obj['class_id']
                color = self.colors[classs_id]

                predicted_class =class_name
                center_y = (top + bottom) / 2.0
                center_x = (left + right) / 2.0
                area = (bottom - top) * (right - left)

                # threshould
                if(score < 0.01):
                    continue

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
    
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[classs_id])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[classs_id])
                draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
                del draw
            
            self.i += 1

            if self.i == 1000:
                self.i = 0
                sys.exit()
     
            self._send_twist(0.2, 0)
            
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

        except CvBridgeError as e:
           print(e)

        #cv2.imshow("Image windowt",img)
        cv2.imshow("Image windowt1", hsv_img)
        cv2.waitKey(3)


    def _send_twist(self, x_linear, z_angular):
        twist = Twist()
        twist.linear.x = float(x_linear)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(z_angular)
        self.pub.publish(twist)


def main(args=None):

    rclpy.init(args=args)
    traveller = DemoYolo()
    try:
        rclpy.spin(traveller)
    finally:
        if traveller not in locals():
            traveller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

