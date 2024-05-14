import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimalBrainDamageDropout(nn.Module):
    def __init__(self, num_features, initial_rate=0.1, max_rate=0.5):
        super(OptimalBrainDamageDropout, self).__init__()
        self.num_features = num_features
        self.initial_rate = initial_rate
        self.max_rate = max_rate
        self.saliency = torch.zeros(num_features, dtype=torch.float)

    def forward(self, x):
        if self.training:
            # Compute dropout probabilities from saliency values
            min_val = self.saliency.min()
            max_val = self.saliency.max()
            normalized_saliency = (self.saliency - min_val) / (max_val - min_val)
            dropout_rates = self.initial_rate + (self.max_grade - self.initial_rate) * (1 - normalized_saliency)
            for i in range(self.num_features):
                mask = torch.bernoulli((1 - dropout_rates[i]).expand_as(x[:, i, :, :])).to(x.device)
                x[:, i, :, :] *= mask
        return x

    def update_saliency(self, model, loss_fn, data_loader):
        model.eval()
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward(create_graph=True)
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    self.saliency += torch.autograd.grad(loss, param, create_graph=True)[0].pow(2).sum().item()
        model.train()

def attach_obd_hooks(layer, obd_dropout_module):
    # This hook updates saliency during training
    def forward_hook(module, input, output):
        obd_dropout_module.update_saliency(module, loss_fn, train_loader)
    layer.register_forward_hook(forward_hook)

