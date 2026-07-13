import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        
        pos_vector = np.arange(0, seq_len).reshape((-1, 1))

        pair_idx = np.arange(0, d_model) // 2 
        dim_idx_vector = 1 / np.power(10000, 2*pair_idx /d_model).reshape((1, -1))
        prod = pos_vector @ dim_idx_vector

        even_idx_mask = np.stack([(np.arange(0, d_model)%2==0).astype(int)]*seq_len)
        odd_idx_mask = np.stack([(np.arange(0, d_model)%2==1).astype(int)]*seq_len)

        return np.round(even_idx_mask * np.sin(prod) + odd_idx_mask * np.cos(prod), 5)
      