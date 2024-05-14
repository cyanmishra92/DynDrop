import torch
import torch.nn.functional as F

class L2DynamicDropout(torch.nn.Module):
    def __init__(self, initial_rate=0.1, max_rate=0.5):
        super(L2DynamicDropout, self).__init__()
        self.initial_rate = initial_rate
        self.max_rate = max_rate

    def forward(self, x, weights):
        if self.training:
            # Calculate the L2 norm of the weights
            l2_norm = torch.norm(weights, p=2, dim=1)  # Compute L2 norm across each filter
            # Normalize the norms to get dropout probabilities
            normalized_l2 = (l2_norm - l2_norm.min()) / (l2_norm.max() - l2_norm.min())
            dropout_rates = self.initial_rate + (self.max_rate - self.initial_rate) * (1 - normalized_l2)
            # Dropout some neurons based on the computed rates
            for i, rate in enumerate(dropout_rates):
                mask = torch.bernoulli(torch.ones_like(x[:, i, :, :]) * (1 - rate)).to(x.device)
                x[:, i, :, :] *= mask
        return x

