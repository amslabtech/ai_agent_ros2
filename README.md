
## Install Ros2
```

$ sudo apt update && sudo apt install curl gnupg2 lsb-release
$ curl http://repo.ros2.org/repos.key | sudo apt-key add -
$ sudo sh -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
$ export CHOOSE_ROS_DISTRO=crystal
$ sudo apt update
$ sudo apt install ros-$CHOOSE_ROS_DISTRO-desktop                     # $CHOOSE_ROS_DISTRO= crystal
$ sudo apt install python3-argcomplete
$ sudo apt install python3-colcon-common-extensions
```

Prepare object_detection
```
cd demo/src/travel/
https://github.com/seqsense/object_detection.git
Prepare package and model by following README in object_detection
```

## Gazebo
```
$ cd gazebo
$ source /opt/ros/crystal/setup.bash
$ gazebo --verbose demo_world56.world
$ killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient
```

## Run
Open another terminal
```
$ cd demo
$ source /opt/ros/crystal/setup.bash
$ colcon build
$ source install/setup.bash && source install/local_setup.bash
```
Run each text_publisher, keyboard_publisher and image_publisher then
```
$ ros2 run travel demo
or
$ ros2 run travel demo_yolo
or
$ ros2 run travel agent
```