import cv2
import rclpy
import numpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class collower(Node):

    def __init__(self):
        super().__init__('follower')
        self.i = 0
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, '/pos/cmd_pos')
        self.sub = self.create_subscription(Image,'/cam/front_camera/image_raw', self.locate)
        self.sub = self.create_subscription(Image,'/cam/left_camera/image_raw', self.publish1)
        self.sub = self.create_subscription(Image,'/cam/right_camera/image_raw', self.publish2)
        #self.timer = self.create_timer(0.1, self.locate)


    def publish1(self,oimg):
        try:
           img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
        except CvBridgeError as e:
           print(e)

        cv2.imshow("Image windowt left", img)
        cv2.waitKey(3)


    def publish2(self,oimg):
        try:
           img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
        except CvBridgeError as e:
           print(e)

        cv2.imshow("Image windowt right", img)
        cv2.waitKey(3)



    def locate(self,oimg):
        try:
           img = self.bridge.imgmsg_to_cv2(oimg, "bgr8")
        except CvBridgeError as e:
           print(e)

        cv2.imshow("Image windowt front", img)
        cv2.waitKey(3)


    def _send_twist(self,z_angular):
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = z_angular
        self.pub.publish(twist)

"""  For sending location info
    def _send_twist(self, x_linear, z_angular):
        twist = Twist()
        twist.linear.x = x_linear
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = z_angular
        self.pub.publish(twist)
"""

def main(args=None):

    rclpy.init(args=args)
    follower = collower()
    try:
        rclpy.spin(follower)
    finally:
        if follower not in locals():
            follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
