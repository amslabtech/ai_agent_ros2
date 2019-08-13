import numpy as np
import os
import argparse
import gym

from simple_dqn import SimpleNetwork, DQN, SARSA, MultiStepDQN, DuelingNetwork

def train_one_episode(env, dqn, episode_num, save_dir, save_episode_interval=20):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    state = np.transpose(env.reset(), (2, 0, 1))
    reward_sum = 0
    frame = 0
    reward = 0
    while True:
        action = dqn.act_and_train(state, reward)
        next_state, reward, done, _ = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))
        if done:
            reward_sum += reward
            state = next_state
            break
        frame += 1
        print('episode:{} frame:{} action:{} reward:{} reward_sum:{}'.format(episode_num, frame, action, reward, reward_sum))
    dqn.stop_episode_and_train(state, reward)
    if save_dir is not None and episode_num % save_episode_interval == 0:
        dqn.save_networks_and_buffer(os.path.join(save_dir, 'episode_{}'.format(episode_num)))
    print('episode:{} reward_sum:{}'.format(episode_num, reward_sum))

def main():
    env = gym.envs.make('Riverraid-v0')
    state = env.reset()
    width, height, channels = state.shape
    action_space = env.action_space.n

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_dir', default=None)

    network_maker = lambda use_gpu: DuelingNetwork(width, height, channels, action_space, use_gpu=use_gpu)
    # network_maker = lambda use_gpu: SimpleNetwork(width, height, channels, action_space, use_gpu=use_gpu)
    dqn = MultiStepDQN(network_maker, action_space=action_space, epsilon=0.2, input_dim=3, replay_start_size=50, buffer_size=500)

    args = vars(parser.parse_args())
    n_episodes = 100
    for i in range(n_episodes):
        train_one_episode(env, dqn, i + 1, args['save_dir'])

if __name__ == '__main__':
    main()