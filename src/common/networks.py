"""Neural network architectures for DQN."""

import torch
import torch.nn as nn
from typing import Tuple


class DQN(nn.Module):
    """Deep Q-Network with configurable hidden layers."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 128),
        activation: str = 'relu'
    ):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function ('relu' or 'tanh')
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


def create_dqn_network(
    state_dim: int,
    action_dim: int,
    num_hidden_layers: int = 3,
    hidden_dim: int = 128
) -> DQN:
    """
    Create a DQN network with specified number of hidden layers.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        num_hidden_layers: Number of hidden layers (3 or 5)
        hidden_dim: Dimension of each hidden layer
        
    Returns:
        DQN network instance
    """
    if num_hidden_layers == 3:
        hidden_dims = (hidden_dim, hidden_dim, hidden_dim)
    elif num_hidden_layers == 5:
        hidden_dims = (hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim)
    else:
        raise ValueError(f"Unsupported number of hidden layers: {num_hidden_layers}. Use 3 or 5.")
    
    return DQN(state_dim, action_dim, hidden_dims)

