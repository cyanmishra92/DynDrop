import torch
import torch.nn.functional as F

class FMREDynamicDropout(torch.nn.Module):
    def __init__(self, initial_rate=0.1, max_rate=0.5, reduction='mean'):
        super(FMREDynamicDropout, self).__init__()
        self.initial_rate = initial_rate
        self.max_rate = max_rate
        self.reduction = reduction
        self.register_buffer('feature_importance', None)

    def forward(self, x):
        if self.training and self.feature_importance is not None:
            # Normalize the feature importance scores to get dropout probabilities
            min_val = self.feature_importance.min()
            max_val = self.feature_importance.max()
            scaled_importance = (self.feature_importance - min_val) / (max_val - min_val)
            dropout_rates = self.initial_rate + (self.max_rate - self.initial_rate) * (1 - scaled_importance)
            # Apply dropout based on importance
            for i, rate in enumerate(dropout_rates):
                mask = torch.bernoulli(torch.ones_like(x[:, i, :, :]) * (1 - rate)).to(x.device)
                x[:, i, :, :] *= mask
        return x

    def update_feature_importance(self, layer_outputs, reconstructed_outputs):
        # Calculate reconstruction error
        if self.reduction == 'mean':
            error = (layer_outputs - reconstructed_outputs).pow(2).mean(dim=[0, 2, 3])
        elif self.reduction == 'sum':
            error = (layer_outputs - reconstructed_outputs).pow(2).sum(dim=[0, 2, 3])
        self.feature_importance = 1 / (error + 1e-5)  # Adding epsilon to avoid division by zero

def attach_fmre_hook(layer, subsequent_layer):
    # Compute the feature map reconstruction error
    def forward_hook(module, input, output):
        # Copy original output for later comparison
        original_output = output.detach()
        # Perform a forward pass of the subsequent layer without this layer's influence
        reconstructed_output = subsequent_layer(output.detach().clone().fill_(0))
        # Update the dropout module of the current layer
        module.dynamic_dropout.update_feature_importance(original_output, reconstructed_output)

    layer.register_forward_hook(forward_hook)

