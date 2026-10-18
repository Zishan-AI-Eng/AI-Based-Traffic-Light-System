import random
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_size):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_size), np.float32)
        self.next_obs = np.zeros((capacity, obs_size), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.idx, self.size = 0, 0

    def add(self, o, a, r, o2, d):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.next_obs[i], self.done[i] = o2, d
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        ids = np.random.randint(0, self.size, batch)
        return (torch.as_tensor(self.obs[ids]), torch.as_tensor(self.act[ids]),
                torch.as_tensor(self.rew[ids]), torch.as_tensor(self.next_obs[ids]),
                torch.as_tensor(self.done[ids]))

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, obs_size, n_actions, lr=1e-3, gamma=0.95,
                 buffer_size=50000, batch_size=64, target_update=250):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.q = QNetwork(obs_size, n_actions)
        self.target = QNetwork(obs_size, n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_size)
        self.learn_steps = 0

    def act(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs).unsqueeze(0))
        return int(q.argmax(1).item())

    def learn(self):
        if len(self.buffer) < self.batch_size * 5:
            return None
        o, a, r, o2, d = self.buffer.sample(self.batch_size)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * self.target(o2).max(1).values * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))