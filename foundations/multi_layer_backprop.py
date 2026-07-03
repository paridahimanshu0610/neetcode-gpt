import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x      = np.array(x)
        y_true = np.array(y_true)  
        W1     = np.array(W1)       
        b1     = np.array(b1)       
        W2     = np.array(W2) 
        b2     = np.array(b2)

        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        loss = np.mean((z2- y_true)**2)
        print(z1.shape, a1.shape, z2.shape)

        n = len(y_true)  if y_true.ndim > 0 else 1
        dL_dz2 = ((2/n) * (z2 - y_true))
        dL_dw2 = (dL_dz2 * a1).reshape(1, -1)
        dL_db2 = dL_dz2

        relu = (z1 > 0).astype(float)
        dL_dz1 = dL_dz2 * W2 * relu
        dL_dw1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1.reshape(-1, )

        loss = np.round(loss, 4)
        dL_dw1, dL_db1, dL_dw2, dL_db2 = np.round(dL_dw1, 4), np.round(dL_db1, 4), np.round(dL_dw2, 4), np.round(dL_db2, 4)

        return {
            'loss': loss,
            'dW1': dL_dw1,
            'db1': dL_db1,
            'dW2': dL_dw2,
            'db2': dL_db2
            }