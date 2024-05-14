import torch
import torch.nn.functional as F

class GradientDynamicDropout(torch.nn.Module):
    def __init__(self, initial_rate=0.1, max_rate=0.5):
        super(GradientDynamicDropout, self).__init__()
        self.initial_rate = initial_rate
        self.max_rate = max_rate
        # We will use this to store the gradient hook
        self.register_buffer('last_gradients', None)

    def forward(self, x):
        if self.training and self.last_gradients is not None:
            # Calculate the magnitude of gradients
            gradient_magnitude = self.last_gradients.norm(p=2, dim=1)
            # Normalize the gradient magnitudes to determine dropout probabilities
            normalized_gradients = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
            dropout_rates = self.initial_rate + (self.max_rate - self.initial_rate) * (1 - normalized_gradients)
            # Apply dropout to the input features
            for i, rate in enumerate(dropout_rates):
                mask = torch.bernoulli(torch.ones_like(x[:, i, :, :]) * (1 - rate)).to(x.device)
                x[:, i, :, :] *= mask
        return x

    def compute_gradient(self, grad):
        self.last_gradients = grad

def attach_gradient_hooks(layer):
    # This function attaches the hooks necessary for the gradient-based dropout
    def hook_fn(grad):
        layer.dynamic_dropout.compute_gradient(grad)
    layer.weight.register_hook(hook_fn)

