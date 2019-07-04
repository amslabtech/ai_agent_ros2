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
from object_detection.mask_rcnn import MaskRCNN
from object_detection.utils import generate_colors, make_r_image
from PIL import Image as PIL_Image
from PIL import ImageDraw
from PIL import ImageFont
import colorsys
import matplotlib.pyplot as plt
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class ObjectDetection(Node):

    def __init__(self):
        super().__init__('traveller')
        self.i = 0
        self.bridge = CvBridge()
        self.pub_obj = self.create_publisher(String, '/demo/objects')
        self.pub_list = []
        self.sub_list = []

        self.model_name = 'mrcnn'

        pub_names = ['/demo/custom_camera/detected_image']
        camera_names = ['/cam/custom_camera/image_raw']

        for pub_name, camera_name in zip(pub_names, camera_names):
            pub = self.create_publisher(Image, pub_name)
            self.pub_list.append(pub)
            self.sub_list.append(self.create_subscription(Image, camera_name, self.locate_closure(pub)))

        PACKAGE_DIR = "src/ai_agent_ros2/travel"
        yolo_data_folder = PACKAGE_DIR+"/"+"object_detection/model_data/yolo3/coco/"
        mrcnn_data_folder = PACKAGE_DIR+"/"+"object_detection/model_data/mrcnn/coco/"
        
        if self.model_name == 'mrcnn':
            classes_path = os.path.join(mrcnn_data_folder, "classes.txt")
            self.model = MaskRCNN(os.path.join(mrcnn_data_folder, "mask_rcnn_coco.h5"), classes_path)
            classes_path = os.path.expanduser(classes_path)
        else:
            yolo_args = {
                "model_path": os.path.join(yolo_data_folder, "yolo.h5"),
                "score": 0.01,
                "anchors_path": os.path.join(yolo_data_folder, "anchors.txt"),
                "classes_path": os.path.join(yolo_data_folder, "classes.txt"),
                    }

            yolo_args = dict((k, v) for k, v in yolo_args.items() if v)
            classes_path = os.path.expanduser(yolo_args["classes_path"])
            self.model = YOLO(**yolo_args)
        
        class_num = 0
        with open(classes_path) as f:
            class_num = len(f.readlines())
        colors = generate_colors(class_num)
        self.colors = colors

    def locate_closure(self, pub):
        def locate(oimg):
            try:
                img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
                hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Object detection
                image = PIL_Image.fromarray(img)

                results = self.model.detect_image(image)

                objects = results['objects']

                for obj in objects:
                    if 'mask' in obj.keys():
                        obj.pop('mask')

                r_image = make_r_image(image, objects, self.colors)
                result = np.asarray(r_image)

                pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
                string = String()
                string.data = json.dumps(results,cls = MyEncoder)
                self.pub_obj.publish(string)

            except CvBridgeError as e:
                print(e)
        return locate

def main(args=None):

    rclpy.init(args=args)
    objectdetection = ObjectDetection()
    try:
        rclpy.spin(objectdetection)
    finally:
        if objectdetection not in locals():
            objectdetection.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()