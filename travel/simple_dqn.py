import os
import random
import pickle
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
from torch.nn import functional as F


Transition = namedtuple('Transicion', ('state', 'action', 'next_state', 'reward'))


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif isinstance(x, list):
        x = torch.Tensor(x)
    assert isinstance(x, torch.Tensor)
    return x


def to_ndarray(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    elif isinstance(x, list):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    return x


class SimpleNetwork(nn.Module):
    def __init__(self, width, height, in_channels, action_num, use_gpu):
        super(SimpleNetwork, self).__init__()
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.action_num = action_num
        self.use_gpu = use_gpu

        self.main_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        out_width = (width // 4 - 1) // 2 - 3
        out_height = (height // 4 - 1) // 2 - 3
        self.fc = nn.Linear(64 * out_width * out_height, 1024)
        self.output = nn.Linear(1024, self.action_num)

    def forward(self, x):
        if self.use_gpu:
            x = x.to('cuda')
        x = to_tensor(x)
        x = self.main_layers(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        x = self.output(x)
        if self.use_gpu:
            x = x.to('cpu')
        return x

class DuelingNetwork(nn.Module):
    def __init__(self, width, height, in_channels, action_num, use_gpu):
        super(DuelingNetwork, self).__init__()
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.action_num = action_num
        self.use_gpu = use_gpu

        self.main_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        mid_width = (width // 2 - 1) // 2 - 3
        mid_height = (height // 2 - 1) // 2 - 3
        self.fc_v = nn.Linear(64 * mid_width * mid_height, 1024)
        self.fc_a = nn.Linear(64 * mid_width * mid_height, 1024)
        self.out_v = nn.Linear(1024, 1)
        self.out_a = nn.Linear(1024, self.action_num)

    def forward(self, x):
        if self.use_gpu:
            x = x.to('cuda')
        x = to_tensor(x)
        x = self.main_layers(x)
        x = x.view(x.size()[0], -1)

        v = F.relu(self.fc_v(x))
        v = F.relu(self.out_v(v))
        a = F.relu(self.fc_a(x))
        a = F.relu(self.out_a(a))

        average = a.mean(1).unsqueeze(1)
        y = v.expand(-1, self.action_num) + a - average
        if self.use_gpu:
            y = y.to('cpu')
        return y

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def add(self, *args):
        add_data = Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(add_data)
        else:
            self.memory[self.index] = add_data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQN:
    def __init__(self, network_maker, action_space, input_dim=3, batch_size=32,
                 gamma=0.99, epsilon=0.2,
                 buffer_size=5000, replay_start_size=500,
                 target_update_interval=10):
        self.input_dim = input_dim
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_count = 0

        self.last_state = None
        self.last_action = None

        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        self.network = network_maker(use_gpu)
        self.target_network = network_maker(use_gpu)

        self.network = self.network.to(device)
        self.target_network = self.target_network.to(device)

        self.replay_start_size = replay_start_size
        self.target_update_interval = target_update_interval

        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action_rewards(self, state):
        state = to_tensor(state)

        if state.ndimension() == self.input_dim:
            state = state[None]
        assert state.ndimension() == self.input_dim + 1

        with torch.no_grad():
            y = self.network(state)[0]
        return to_ndarray(y)

    def _update(self, x, y):
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(to_tensor(x), to_tensor(y))
        loss.backward()
        self.optimizer.step()
    
    def _train(self, data_batch):
        assert len(data_batch) == self.batch_size
        if self.train_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.eval()
        self.train_count += 1
        batch = Transition(*zip(*data_batch))
        state_batch = torch.stack(list(batch.state))
        next_state_batch = torch.stack(list(batch.next_state))
        reward_batch = torch.Tensor(batch.reward)
        action_batch = torch.Tensor(batch.action).view(-1, 1).long()

        predicted_result = self.network(state_batch)
        predicted_values = predicted_result.gather(1, action_batch).squeeze()

        not_done = np.array(list(map(lambda x: x.next_state is not None, data_batch)), dtype=np.uint8)
        next_states = torch.Tensor(next_state_batch[not_done])
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[not_done] = self.target_network(next_states).max(1)[0].detach()
        excepted_values = (next_state_values * self.gamma) + reward_batch
        self._update(predicted_values, excepted_values)

    def add_data(self, state, action, next_state, reward, train=True):
        state = to_tensor(state)
        next_state = to_tensor(next_state)
        self.replay_buffer.add(state, action, next_state, reward)
        if train and len(self.replay_buffer.memory) >= self.replay_start_size:
            self._train(self.replay_buffer.sample(self.batch_size))

    def act_and_train(self, state, last_reward):
        action = self.act(state)

        if self.last_state is not None:
            assert self.last_action is not None
            self.add_data(self.last_state, self.last_action, state, last_reward)

        self.last_state = state
        self.last_action = action

        return action

    def stop_episode_and_train(self, state, last_reward):
        if self.last_state is not None:
            assert self.last_action is not None
            self.add_data(self.last_state, self.last_action, state, last_reward)
        self.last_state = None
        self.last_action = None

    def act(self, state, test=False):
        if not test and (random.random() < self.epsilon):
            return np.random.randint(self.action_space)
        rewards = self.get_action_rewards(state)
        return np.argmax(rewards)

    def save_network(self, path):
        torch.save(self.network, path)

    def save_networks_and_buffer(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        # TODO: save namedtuple
        '''
        with open(os.path.join(dir_path, 'replay_buffer.pkl'), 'w') as f:
            pickle.dump(self.replay_buffer, f)
            '''
        torch.save(self.network, os.path.join(dir_path, 'network.pth'))
        torch.save(self.target_network, os.path.join(dir_path, 'target_network.pth'))

    def load_network(self, path):
        self.network = torch.load(path)
        self.network.eval()

    def load_networks_and_buffer(self, dir_path):
        '''
        with open(os.path.join(dir_path, 'replay_buffer.pkl'), 'r') as f:
            self.replay_buffer = pickle.load(f)
            '''
        self.network = torch.load(os.path.join(dir_path, 'network.pth'))
        self.target_network = torch.load(os.path.join(dir_path, 'target_network.pth'))
        self.network.eval()
        self.target_network.eval()


class MultiStepDQN(DQN):
    def __init__(self, network_maker, action_space, n_steps=3, input_dim=3, batch_size=32,
                 gamma=0.99, epsilon=0.2,
                 buffer_size=5000, replay_start_size=500,
                 target_update_interval=10):
        super(MultiStepDQN, self).__init__(network_maker, action_space, input_dim, batch_size,
                                  gamma, epsilon, buffer_size, replay_start_size, target_update_interval)
        delattr(self, 'last_state')
        delattr(self, 'last_action')
        self.n_steps = n_steps
        self.state_queue = deque()
        self.action_queue = deque()
        self.reward_queue = deque()
        self.step_gamma = gamma
        self.gamma **= n_steps

    def pop_and_add_data(self, state):
        if len(self.state_queue) > self.n_steps:
            origin_state = self.state_queue.popleft()
            origin_action = self.action_queue.popleft()
            self.reward_queue.popleft()
            reward_sum = 0
            reward_per = 1.0
            for r in self.reward_queue:
                reward_sum += r * reward_per
                reward_per *= self.step_gamma
            self.add_data(origin_state, origin_action, state, reward_sum)

    def act_and_train(self, state, last_reward):
        action = self.act(state)

        self.state_queue.append(state)
        self.action_queue.append(action)
        self.reward_queue.append(last_reward)
        self.pop_and_add_data(state)

        self.last_state = state
        self.last_action = action

        return action

    def stop_episode_and_train(self, state, last_reward):
        self.state_queue.append(state)
        self.reward_queue.append(last_reward)
        self.pop_and_add_data(state)
        self.state_queue.clear()
        self.action_queue.clear()
        self.reward_queue.clear()


class SARSA(DQN):
    def act(self, state, test=False):
        if not test and (random.random() < self.epsilon):
            return np.random.randint(self.action_space)
        rewards = self.get_action_rewards(state)
        min_val = np.min(rewards)
        if min_val < 0:
            rewards += abs(min_val)
        if np.equal(np.sum(rewards), 0.0):
            rewards += 1.0
        rewards /= np.sum(rewards)
        return np.random.choice(np.arange(rewards.size), p=rewards.flatten())