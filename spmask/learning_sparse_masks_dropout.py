import torch
import torch.nn as nn
import torch.nn.functional as F

class LearningSparseMasksDropout(nn.Module):
    def __init__(self, channel_size, sparsity_target=0.5, learning_rate=1e-3, initial_rate=None, max_rate=None):
        super(LearningSparseMasksDropout, self).__init__()
        # Store the parameters
        self.sparsity_target = sparsity_target
        self.learning_rate = learning_rate
        # These parameters are accepted for interface compatibility but not used
        self.initial_rate = initial_rate
        self.max_rate = max_rate

        # Initialize dropout masks as trainable parameters
        self.masks = nn.Parameter(torch.ones(channel_size))
        self.optimizer = torch.optim.Adam([self.masks], lr=self.learning_rate)

    def forward(self, x):
        if self.training:
            # Apply a sigmoid to get dropout probabilities between 0 and 1
            probabilities = torch.sigmoid(self.masks)
            # Sample a Bernoulli distribution to create a binary mask
            binary_mask = torch.bernoulli(probabilities).to(x.device)
            # Apply the mask
            x = x * binary_mask.unsqueeze(-1).unsqueeze(-1)
            # Optionally, apply a scaling factor to maintain the mean activation value
            x = x / (probabilities.unsqueeze(-1).unsqueeze(-1) + 1e-8)
        return x

    def update_masks(self):
        # Calculate and apply gradients to minimize the mask sparsity loss
        sparsity_loss = F.mse_loss(torch.sigmoid(self.masks), torch.full_like(self.masks, self.sparsity_target))
        self.optimizer.zero_grad()
        sparsity_loss.backward()
        self.optimizer.step()

