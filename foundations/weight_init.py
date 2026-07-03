import torch
import torch.nn as nn
import math
from typing import List

# Explanation for why we multiply by a factor which is related to the previous and next units:
# https://claude.ai/chat/9f1fda00-b3ac-49ca-8cf2-175eddee013f

# Short Explanation:
# Our goal is to preserve the variance of input and out at any layer to prevent exploding/vanishing issue.
# Without this initialization, for a single unit: var(wi * xi) = var(wi) * var(xi) = var(xi) [Assuming var(wi) = 1 for random initialization]
# So, for all 'n' previous layer units, total variance of output = n * var(xi)
# If we don't multiply it by a factor of 1/n, the variance will grow with the number of units.

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2)/(math.sqrt(fan_in + fan_out))
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2)/(math.sqrt(fan_in))
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        prev_units, next_units = input_dim, hidden_dim
        weights = []

        for _ in range(num_layers):
            if init_type == "xavier":
                std =  math.sqrt(2/(prev_units + next_units))
            elif init_type == "kaiming":
                std = math.sqrt(2/prev_units)
            else:
                std = 1
            weights.append(torch.randn((next_units, prev_units))* std)
            prev_units, next_units = hidden_dim, hidden_dim

        x = torch.randn((1, input_dim))
        res = []
        for wt in weights:
            x = x @ wt.T
            x = torch.relu(x)
            res.append(round(x.std().item(), 2))

        return res