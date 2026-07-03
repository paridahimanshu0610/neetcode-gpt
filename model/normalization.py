import numpy as np
from numpy.typing import NDArray

# Explanation for each normalization type:
# https://claude.ai/chat/3f276c6b-e330-4a5a-94d7-8d9b7f3d7a67

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean, var = x.mean(), np.var(x)
        eps = 1e-5 
        x_hat = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

        return np.round(out, 5)