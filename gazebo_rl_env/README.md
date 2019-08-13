# GazeboRLEnv

Environment for RL simulation in Gazebo with gym.

# Usage

```
# Move to top pkg dir
cd ~/ws/src/ai_agent_ros2/gazebo_rl_env
# Installation
pip3 install -e .

# build ros2 pkgs
cd ~/ws
colcon build
# source related pkg and set path
r2; rs;
source src/ai_agent_ros2/setup.sh 

'''
# Execute program in GUI 
cd ~/ws/src/ai_agent_ros2
python3 scripts/gg_demo.py -g
python3 scripts/gg_demo_using_dqn.py -g
```
