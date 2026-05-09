import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01

    def push(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        transition = (state, action, reward, next_state, done)
        self.tree.add(max_p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), idxs, is_weight

    def update_priorities(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (np.abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), None, None

    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, in_dim=192, hid_dim=256, num_layers=1, cell='gru', dueling=False):
        super(DQNetwork, self).__init__()
        self.dueling = dueling
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
            
        if self.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 2)
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 2)
            )

    def forward(self, x, return_hidden=False):
        h, _ = self.rnn(x)
        q = self.forward_q(h)
        if return_hidden:
            return q, h
        return q

    def forward_q(self, h):
        if self.dueling:
            v = self.value_stream(h)
            a = self.advantage_stream(h)
            q = v + a - a.mean(dim=-1, keepdim=True)
        else:
            q = self.q_head(h)
        return q

class DQNAgent:
    def __init__(self, in_dim=192, hid_dim=256, num_layers=1, cell='gru',
                 lr=1e-4, gamma=0.99, tau=0.005, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=500, batch_size=64, memory_size=10000,
                 use_double=False, use_dueling=False, use_per=False, n_step=1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_double = use_double
        self.use_per = use_per
        self.n_step = n_step
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.online_net = DQNetwork(in_dim, hid_dim, num_layers, cell, use_dueling).to(self.device)
        self.target_net = DQNetwork(in_dim, hid_dim, num_layers, cell, use_dueling).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)

    def select_action(self, h, training=True):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if training:
            self.steps_done += 1

        seq_len = h.size(1)
        if training and sample < eps_threshold:
            action = torch.randint(0, 2, (1, seq_len), device=self.device)
        else:
            with torch.no_grad():
                q_values = self.online_net.forward_q(h) 
                action = q_values.argmax(dim=-1)
                
        return action

    def soft_update(self):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        if self.use_per:
            is_weights = torch.FloatTensor(is_weights).to(self.device).unsqueeze(-1)

        q_values = self.online_net.forward_q(states)
        q_value = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.online_net.forward_q(next_states).argmax(dim=-1, keepdim=True)
                next_q_values = self.target_net.forward_q(next_states)
                next_q_value = next_q_values.gather(-1, next_actions).squeeze(-1)
            else:
                next_q_values = self.target_net.forward_q(next_states)
                next_q_value = next_q_values.max(dim=-1)[0]
                
        gamma_n = self.gamma ** self.n_step
        expected_q_value = rewards + gamma_n * next_q_value * (1 - dones)

        loss_fn = nn.SmoothL1Loss(reduction='none' if self.use_per else 'mean')
        loss = loss_fn(q_value, expected_q_value)

        if self.use_per:
            errors = torch.abs(q_value - expected_q_value).detach().cpu().numpy()
            self.memory.update_priorities(idxs, errors)
            loss = (loss * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5.0) 
        self.optimizer.step()

        self.soft_update()

        return loss.item()
