import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Critic Class - Defines the neural network for the critic model in reinforcement learning.
class Critic(nn.Module):
    # Initializer for the Critic class.
    def __init__(self, input_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)  # Set a seed for reproducibility.
        
        # Define fully connected layers. The network has five linear layers with decreasing units.
        self.fc1 = nn.Linear(input_size, 256)  # First fully connected layer from input to 256 units.
        self.fc2 = nn.Linear(256, 128)         # Second layer: 256 to 128 units.
        self.fc3 = nn.Linear(128, 64)          # Third layer: 128 to 64 units.
        self.fc4 = nn.Linear(64, 32)           # Fourth layer: 64 to 32 units.
        self.fc5 = nn.Linear(32, 1)            # Final layer to output a single value (Q-value).

        self.bn1 = nn.BatchNorm1d(256)         # Batch normalization for the first layer's output.

        self.reset_parameters()                # Initialize network weights.

    # Forward pass function. Defines how the combined state and action is processed to produce Q-values.
    def forward(self, states, actions):
        # Concatenate states and actions as input.
        x_state_action = torch.cat((states, actions), dim=1)
        
        # Apply layers with ReLU activations, except for the final layer.
        x = F.relu(self.fc1(x_state_action))  # ReLU activation for the first layer.
        x = self.bn1(x)                       # Batch normalization.
        x = F.relu(self.fc2(x))               # ReLU activation for the second layer.
        x = F.relu(self.fc3(x))               # ReLU activation for the third layer.
        x = F.relu(self.fc4(x))               # ReLU activation for the fourth layer.
        x = self.fc5(x)                       # Output layer without activation (Q-value).

        return x

    # Function to reset the weights of the network for training stability.
    def reset_parameters(self):
        # Initialize weights for each layer using a uniform distribution based on layer's fan-in.
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

# Function to calculate the initialization range based on the number of input units to a layer.
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]  # Number of input units.
    lim = 1. / np.sqrt(fan_in)            # Calculation of the limit for uniform distribution.
    return (-lim, lim)
