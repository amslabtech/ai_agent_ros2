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
import time
 
class Text(Node):

    def __init__(self):
        super().__init__('text')
        self.pub = self.create_publisher(String, '/condition_text')
        self.current_text = ""
        self.check()



    def check(self):
        while True:
            f = open('condition.txt')
            text_state = f.read()
            if self.current_text != text_state:
                self.current_text = text_state
                string = String()
                string.data = text_state.strip()
                self.pub.publish(string)
            time.sleep(0.05)


def main(args=None):

    rclpy.init(args=args)
    text = Text()
    try:
        rclpy.spin(text)
    finally:
        if text not in locals():
            text.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
