import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        rms = np.sqrt(np.square(x).mean() + eps)

        x_hat = x / rms
        out = np.round(gamma * x_hat, 4)

        return out.tolist()  
