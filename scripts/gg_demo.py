""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gazeborlenv
import time

env = gym.make('AICar-v0')

sample_a = env.action_space.sample()
assert sample_a < 3, "action id error: {}".format(sample_a)

while True:
    # take a random action
    sample_a = env.action_space.sample()
    observation, reward, done, info = env.step(sample_a)
    print("(obs,r,d,info) = ({},{},{},{})".format(observation,reward,done,info))
