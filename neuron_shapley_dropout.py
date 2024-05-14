import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronShapleyDropout(nn.Module):
    def __init__(self, num_features, sample_size=100, sparsity_target=0.5):
        super(NeuronShapleyDropout, self).__init__()
        self.num_features = num_features
        self.sample_size = sample_size
        self.sparsity_target = sparsity_target
        self.shapley_values = nn.Parameter(torch.zeros(num_features), requires_grad=False)

    def forward(self, x):
        if self.training:
            # Scale Shapley values to [0, 1] and compute dropout probabilities
            min_val = self.shapley_values.min()
            max_val = self.shapley_values.max()
            scaled_values = (self.shapley_values - min_val) / (max_val - min_val + 1e-5)
            dropout_rates = self.sparsity_target + (1 - self.sparsity_target) * (1 - scaled_values)
            for i in range(self.num_features):
                mask = torch.bernoulli((1 - dropout_rates[i]).expand_as(x[:, i, :, :])).to(x.device)
                x[:, i, :, :] *= mask
        return x

    def update_shapley_values(self, model, loss_fn, input_data, target):
        with torch.no_grad():
            baseline_loss = loss_fn(model(input_data), target)
            for i in range(self.num_features):
                # Create a copy of the model for each feature
                model_copy = copy.deepcopy(model)
                # Remove the contribution of the i-th feature
                if isinstance(model_copy, nn.Conv2d):
                    model_copy.weight.data[:, i, :, :] = 0
                    model_copy.bias.data[i] = 0
                elif isinstance(model_copy, nn.Linear):
                    model_copy.weight.data[i, :] = 0
                    model_copy.bias.data[i] = 0
                perturbed_loss = loss_fn(model_copy(input_data), target)
                marginal_contribution = baseline_loss - perturbed_loss
                self.shapley_values[i] += marginal_contribution / self.sample_size

