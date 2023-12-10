import torch as T
from torch import nn
import torch.optim as optim
import numpy as np
import pickle

class Critic(nn.Module):
    def __init__(self, input_dims, alpha, checkpoint='critic_net_ppo') -> None:
        super().__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.network = nn.Sequential(
            nn.Linear(*input_dims,256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_tensor = T.Tensor(state).to(self.device)
        value = self.network(state_tensor)
        return value

    def save(self):
        with open(self.checkpoint, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def load(self):
        with open(self.checkpoint, 'rb') as f:
            self.load_state_dict(pickle.load(f))