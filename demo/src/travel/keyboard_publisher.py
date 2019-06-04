import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pynput.keyboard import Key, Listener
import time
 
class Keyboard(Node):

    def __init__(self):
        super().__init__('text')
        self.pub = self.create_publisher(String, '/keyboard')
        self.check()

    def on_press(self, key):
        pub_str = String()
        pub_str.data = str(key)
        self.pub.publish(pub_str)

    def on_release(self, key):
        pub_str = String()
        pub_str.data = 'None'
        self.pub.publish(pub_str)
        if key == Key.esc:
            # Stop listener
            return False

    def check(self):
        with Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()


def main(args=None):

    rclpy.init(args=args)
    text = Keyboard()
    try:
        rclpy.spin(text)
    finally:
        if text not in locals():
            text.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
