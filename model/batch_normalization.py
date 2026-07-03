import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        eps = 1e-5

        if training:
            mean = np.mean(x, axis = 0)
            var = np.mean(np.square(x - mean), axis = 0)
            running_mean = (1-momentum) * running_mean + momentum * mean
            running_var = (1-momentum) * running_var + momentum * var
            x_hat = (x - mean) / np.sqrt(var + eps)
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps) 

        out = gamma * x_hat + beta

        out, running_mean, running_var = np.round(out, 4), np.round(running_mean, 4), np.round(running_var, 4)
          
        return out.tolist(), running_mean.tolist(), running_var.tolist()  