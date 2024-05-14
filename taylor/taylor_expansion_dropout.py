import torch
import torch.nn.functional as F

class TaylorExpansionDropout(torch.nn.Module):
    def __init__(self, initial_rate=0.1, max_rate=0.5):
        super(TaylorExpansionDropout, self).__init__()
        self.initial_rate = initial_rate
        self.max_rate = max_rate
        self.register_buffer('last_activation', None)
        self.register_buffer('last_gradient', None)

    def forward(self, x):
        if self.training and self.last_activation is not None and self.last_gradient is not None:
            # Calculate the Taylor Expansion saliency
            saliency = (self.last_activation * self.last_gradient).abs().sum(dim=(2, 3))
            # Normalize the saliency to determine dropout probabilities
            normalized_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            dropout_rates = self.initial_rate + (self.max_rate - self.initial_rate) * (1 - normalized_saliency)
            # Apply dropout to the input features
            for i, rate in enumerate(dropout_rates):
                mask = torch.bernoulli(torch.ones_like(x[:, i, :, :]) * (1 - rate)).to(x.device)
                x[:, i, :, :] *= mask
        return x

    def compute_activation_gradient(self, activation, grad_output):
        self.last_activation = activation
        self.last_gradient = grad_output

def attach_taylor_hooks(layer):
    # This function attaches the hooks necessary for the Taylor-based dropout
    def forward_hook(module, input, output):
        module.dynamic_dropout.last_activation = output.detach()

    def backward_hook(module, grad_input, grad_output):
        module.dynamic_dropout.compute_activation_gradient(module.output, grad_output[0])

    layer.register_forward_hook(forward_hook)
    layer.weight.register_backward_hook(backward_hook)

