import torch
import torch.nn as nn
import torch.nn.functional as F
from configparser import ConfigParser
from dropout_factory import get_dropout_method

# Load configuration settings
config = ConfigParser()
config.read('config.ini')

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input: 1x28x28, Output: 6x24x24 (after convolution), Output: 6x12x12 (after pooling)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input: 6x12x12, Output: 16x8x8 (after convolution), Output: 16x4x4 (after pooling)

        # Define the fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten 16x4x4 to create 256 input features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for MNIST

        # Read dropout configuration and initialize dropout layer
        dropout_method_name = config['Dropout']['Method']
        dropout_params = {
            'channel_size': 64,  # Assuming dropout is used before fc3 which has 84 inputs
            'initial_rate': float(config['Dropout']['InitialRate']),
            'max_rate': float(config['Dropout']['MaxRate']),
            'sparsity_target': 0.5,  # Default value, modify as needed
            'learning_rate': 0.01  # Default learning rate for masks or other parameters, modify as needed
        }
        dropout_class = get_dropout_method(dropout_method_name)
        self.dropout = dropout_class(**dropout_params)

    def forward(self, x):
        # Convolutional and pooling layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flattening the output for the fully connected layers
        x = x.view(-1, 16 * 4 * 4)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

