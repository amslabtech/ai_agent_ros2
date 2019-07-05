cd `dirname $0`

. /usr/share/gazebo/setup.sh
. ~/ws/install/setup.bash
export GAZEBO_RESOURCE_PATH=~/ws/src/ai_agent_ros2/gazebo/worlds:${GAZEBO_RESOURCE_PATH}
export GAZEBO_MODEL_PATH=~/ws/src/ai_agent_ros2/gazebo/models:${GAZEBO_MODEL_PATH}
