
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

Create model file using
https://github.com/amslabtech/object_detection
and copy it to 
gazebo/src/travel/src/yolo/

## Gazebo
```
$ cd gazebo
$ gazebo --verbose demo_world56.world
```

## Run in Gazebo
Open another terminal
```
$ cd gazebo
$ source /opt/ros/crystal/setup.bash
$ colcon build
$ source install/setup.bash && source install/local_setup.bash
$ ros2 run travel demo
or
$ ros2 run travel demo_yolo
$ killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient
```