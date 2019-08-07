from cv_bridge import CvBridge
import gym
gym.logger.set_level(40)  # hide warnings
import numpy as np
import copy
import psutil
from gym import spaces
from gazeborlenv.utils import ut_generic, ut_launch
from gym.utils import seeding
import subprocess
from PIL import Image as PIL_Image

# ROS 2
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Empty
from ros2pkg.api import get_prefix_path


class AICarEnv(gym.Env):
    """
    TODO. Define the environment.
    """

    def __init__(self):
        """
        Initialize the AICar environemnt
        """
        # Manage command line args
        args = ut_generic.getArgsParser().parse_args()
        self.gzclient = args.gzclient
        self.multiInstance = args.multiInstance
        self.port = args.port

        # Launch simulation in a new Process
        self.launch_subp = ut_launch.startLaunchServiceProcess(
            ut_launch.generateLaunchDescription(
                self.gzclient, self.multiInstance, self.port))

        # Create the node after the new ROS_DOMAIN_ID is set in
        # generate_launch_description()
        rclpy.init(args=None)
        self.node = rclpy.create_node(self.__class__.__name__)

        # class variables
        # self._observation_msg = None
        self._observation_img = None
        self.max_episode_steps = 1024  # default value, can be updated from baselines
        self.iterator = 0
        self.reset_flag = True

        # ai_agent
        self.pub = self.node.create_publisher(String, '/pos/action_id')
        camera_names = ['/cam/custom_camera/image_raw']
        self.sub_img = self.node.create_subscription(
            Image, camera_names[0], self.observation_img_callback)
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')

        # 0: "forward", 1: "left", 2: "right"
        self.action_space = gym.spaces.Discrete(3)

        # observation = (240,320,3)
        screen_height, screen_width = (240, 320)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(
                screen_height, screen_width, 3), dtype=np.uint8)

        self.bridge = CvBridge()

    def observation_img_callback(self, oimg):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_img = oimg

    def set_episode_size(self, episode_size):
        """
        Set max episode size
        """
        self.max_episode_steps = episode_size

    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # # Take an observation
        rclpy.spin_once(self.node)
        obs_img = self._observation_img

        # Check that the observation is not prior to the action
        while obs_img is None:
            rclpy.spin_once(self.node)
            obs_img = self._observation_img

        img = self.bridge.imgmsg_to_cv2(obs_img, "bgr8")
        img = PIL_Image.fromarray(img)
        obs_img = np.asarray(img, dtype="uint8")

        self._observation_img = None

        return obs_img

    def computeReward(self):
        """
        Compute reward
        """
        ## To Do: implementation
        return -1

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - action
            - observation
            - reward
            - done (status)
        """
        self.iterator += 1

        # execute action
        id_msg = String()
        action_id = action
        id_msg.data = str(action_id)  # Publish given action_id by argument
        self.pub.publish(id_msg)

        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        # Take an observation
        obs = self.take_observation()

        # Calc reward
        reward = self.computeReward()

        # Calculate if the env has been solved
        done = bool(self.iterator == self.max_episode_steps)

        ## To Do: implementation
        info = {}

        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, info

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        if self.reset_flag is True:
            # reset simulation
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)

        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        # Take an observation
        obs = self.take_observation()

        # Return the corresponding observation
        return obs

    def close(self):
        print("Closing " + self.__class__.__name__ + " environment.")
        parent = psutil.Process(self.launch_subp.pid)
        for child in parent.children(recursive=True):
            child.kill()
        rclpy.shutdown()
        parent.kill()
