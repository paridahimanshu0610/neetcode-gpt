import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.inference_mode():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = torch.mean(x).item()
                    std = torch.std(x).item()
                    if x.dim() >= 2:
                        dead_fraction = (((x <= 0).all(dim=0)).float().mean()).item()
                    else:
                        dead_fraction = ((x <= 0).float().mean()).item()

                    mean, std, dead_fraction = round(mean, 4), round(std, 4), round(dead_fraction, 4)
                    res.append({'mean':mean, 'std':std, 'dead_fraction':dead_fraction})

        return res 

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})
        
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for activation_stat in activation_stats:
            if activation_stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for gradient_stat in gradient_stats:
            if gradient_stat['norm'] > 1000:
                return 'exploding_gradients'
            elif gradient_stat['norm'] < 1e-5:
                return 'vanishing_gradients'

        for activation_stat in activation_stats:
            if (activation_stat['std'] > 10.0):
                return 'exploding_gradients'
            elif (activation_stat['std'] < 0.1):
                return 'vanishing_gradients'
        
        return 'healthy'

