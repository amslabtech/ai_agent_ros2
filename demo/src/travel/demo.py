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
from PIL import Image as PIL_Image
from PIL import ImageDraw
from PIL import ImageFont


class Action():
    
    def __init__(self):
        self.twists = []

class Demo(Node):

    def __init__(self):
        super().__init__('traveller')
        self.i = 0
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        self.sub = self.create_subscription(Image,'/cam/custom_camera/image_raw', self.locate)

        self.objects = []
        self.now_state = 0
        self.actions = []

    def locate(self,oimg):
        try:
            img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
            hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image=PIL_Image.fromarray(img)

            # Object detection
            

            self.i += 1

            if self.i == 1000:
                self.i = 0
                sys.exit()

            self._send_twist(-0.2, 0)

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
    traveller = Demo()
    try:
        rclpy.spin(traveller)
    finally:
        if traveller not in locals():
            traveller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
