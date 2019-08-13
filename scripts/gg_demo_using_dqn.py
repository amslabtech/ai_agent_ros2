""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import numpy as np
import gazeborlenv
import time
import os
import sys

# for dqn
sys.path.append(os.path.join(os.path.dirname(__file__), '../travel/'))
from simple_dqn import SimpleNetwork, DQN


# save_dir = None
save_dir = './train_results'

load_episode_dir_path = None
# load_episode_dir_path= './train_results/episode_10'


episode_num = 1000
save_episode_interval = 10

env = gym.make('AICar-v0')

observation = np.transpose(env.reset(), (2, 0, 1))

channels, width, height = observation.shape

network_maker = lambda use_gpu: SimpleNetwork(width, height, channels, env.action_space.n, use_gpu=use_gpu)
dqn = DQN(network_maker, action_space=env.action_space.n, epsilon=0.2, input_dim=3, replay_start_size=50, buffer_size=500)

if save_dir is not None:
    os.makedirs(save_dir, exist_ok=True)

if load_episode_dir_path is not None:
    dqn.load_networks_and_buffer(load_episode_dir_path)
    print('model loaded from {}'.format(load_episode_dir_path))

for i in range(episode_num):
    observation = np.transpose(env.reset(), (2, 0, 1))
    reward_sum = 0
    reward = 0

    while True:
        action = dqn.act_and_train(observation, reward)
        next_observation, reward, done, info = env.step(action)
        next_observation = np.transpose(next_observation, (2, 0, 1))
        reward_sum += reward
        if done:
            dqn.stop_episode_and_train(next_observation, reward)
            break

        observation = next_observation
    print('episode finished   reward sum:{}'.format(reward_sum))
    if save_dir is not None and (i + 1) % save_episode_interval == 0:
        save_path = os.path.join(save_dir, 'episode_{}'.format(i + 1))
        print('model saved to {}'.format(save_path))
        dqn.save_networks_and_buffer(save_path)