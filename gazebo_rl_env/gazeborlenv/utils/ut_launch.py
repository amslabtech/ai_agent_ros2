import socket
import random
import os
import pathlib

from datetime import datetime
from billiard import Process

from ament_index_python.packages import get_package_prefix
from launch import LaunchService, LaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch_ros.actions import Node

import gazeborlenv
from gazeborlenv.utils import ut_generic


def startLaunchServiceProcess(launchDesc):
    """Starts a Launch Service process. To be called from subclasses.

    Args:
         launchDesc : LaunchDescription obj.
    """
    # Create the LauchService and feed the LaunchDescription obj. to it.
    launchService = LaunchService()
    launchService.include_launch_description(launchDesc)
    process = Process(target=launchService.run)
    # The daemon process is terminated automatically before the main program exits,
    # to avoid leaving orphaned processes running
    process.daemon = True
    process.start()

    return process


def generateLaunchDescription(gzclient, multiInstance, port):
    """
        Returns ROS2 LaunchDescription object.
    """
    try:
        envs = {}
        for key in os.environ.__dict__["_data"]:
            key = key.decode("utf-8")
            if key.isupper():
                envs[key] = os.environ[key]
    except BaseException as exception:
        print("Error with Envs: " + str(exception))
        return None

    # Gazebo visual interfaze. GUI/no GUI options.
    if gzclient:
        gazeboCmd = "gazebo"
    else:
        gazeboCmd = "gzserver"

    # Creation of ROS2 LaunchDescription obj.

    worldPath = os.path.join(os.path.dirname(gazeborlenv.__file__), 'worlds',
                             'test8.world')
    '''
    worldPath = os.path.join(os.path.dirname(gazeborlenv.__file__), 'worlds',
                             'empty.world')
                             '''

    launchDesc = LaunchDescription([
        ExecuteProcess(
            cmd=[gazeboCmd, '--verbose', '-s', 'libgazebo_ros_factory.so', '-s',
                 'libgazebo_ros_init.so', worldPath], output='screen', env=envs),
        Node(package='travel', node_executable='spawn_agent',
             output='screen'),
        Node(package='travel', node_executable='agent',
             output='screen')
    ])
    return launchDesc
