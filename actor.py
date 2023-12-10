import torch as T
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pickle

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, checkpoint='actor_net_ppo') -> None:
        super().__init__()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        # red neuronal para policy
        self.checkpoint = checkpoint
        self.network = nn.Sequential(
            nn.Linear(*input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_tensor = T.Tensor(state).to(self.device)
        dist = self.network(state_tensor)
        action_pd = T.distributions.Categorical(dist)
        return action_pd        

    def save(self):        
        with open(self.checkpoint, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def load(self):
        with open(self.checkpoint, 'rb') as f:
            self.load_state_dict(pickle.load(f))