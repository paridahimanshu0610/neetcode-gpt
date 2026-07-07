import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        def compute_loss(y_true, y_pred):
            return np.average((y_true - y_pred)**2, axis = 0)
        
        def compute_gradient(cache, y_true):
            X, z1, a1 = cache        
            m = X.shape[0]
            
            dz1 = (2/m) * (a1-y_true)
            dw1 = X.T @ dz1
            db1 = np.sum(dz1, axis = 0, keepdims = True)

            return {"dw1":dw1, "db1":db1}

        def forwardpass(X, wt, b):
            z1 = X @ wt + b
            a1 = z1

            return z1, a1

        def update_params(wt, b, grad, lr):
            wt = wt - lr * grad["dw1"]
            b = b - lr * grad["db1"]

            return wt, b

        y = y.reshape((-1,1))
        n = X.shape[1] # Number of features

        w1 = np.zeros((n,1))
        b1 = np.zeros((1,1))

        for epoch in range(epochs):
            z1, a1 = forwardpass(X, w1, b1)
            loss = compute_loss(y, a1)
            grad = compute_gradient((X, z1, a1), y)
            w1, b1 = update_params(w1, b1, grad, lr)

        w1 = np.round(w1.reshape((-1,)), 5)
        b1 = np.round(b1[0,0], 5).item()

        return w1, b1