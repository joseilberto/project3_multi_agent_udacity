import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, nhidden = [128, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            nhidden (list): Number of nodes in each hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.nhidden = nhidden
        
        self.fc1 = nn.Linear(state_size, nhidden[0])
        self.fc2 = nn.Linear(nhidden[0], nhidden[1])
        self.fc3 = nn.Linear(nhidden[1], action_size)

        self.batch_norm = nn.BatchNorm1d(nhidden[0])

        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1: 
            state = torch.unsqueeze(state, 0)
        x = state.clone()
        x = self.activation(self.fc1(x))
        x = self.batch_norm(x)
        x = self.activation(self.fc2(x))        
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, nhidden = [128, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            nhidden (list): Number of nodes in each hidden layer
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.nhidden = nhidden
        
        self.fc1 = nn.Linear(state_size, nhidden[0])
        self.fc2 = nn.Linear(nhidden[0] + action_size, nhidden[1])
        self.fc3 = nn.Linear(nhidden[1], 1)

        self.batch_norm = nn.BatchNorm1d(nhidden[0])

        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""        
        if state.dim() == 1: 
            state = torch.unsqueeze(state, 0)
        x = state.clone()
        x = self.activation(self.fc1(x))
        x = self.batch_norm(x)
        x = torch.cat((x, action), dim=1)
        x = self.activation(self.fc2(x))
        return self.fc3(x)