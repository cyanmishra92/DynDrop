import torch.nn as nn
import torch.nn.functional as F
from configparser import ConfigParser
from dropout_factory import get_dropout_method

# Load configuration
config = ConfigParser()
config.read('config.ini')

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.NextLinear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        # Initialize dropout modules for all layers with the same configuration
        dropout_method_name = config['Dropout']['Method']
        dropout_params = {
            'initial_rate': float(config['Dropout']['InitialRate']),
            'max_rate': float(config['Dropout']['MaxRate'])
        }
        dropout_module = get_dropout_method(dropout_method_name, **dropout_params)

        # Attach the same dynamic dropout module to all applicable layers
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            setattr(layer, 'dynamic_dropout', dropout_module)

    def forward(self, x):
        x = self._apply_dropout(self.conv1, x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self._apply_dropout(self.conv2, x)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self._apply_dropout(self.fc1, x)
        x = F.relu(x)
        x = self._apply_dropout(self.fc2, x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def _apply_dropout(self, layer, x):
        if hasattr(layer, 'dynamic_dropout'):
            return layer.dynamic_dropout(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

