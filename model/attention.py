import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.key_layer = nn.Linear(in_features = embedding_dim, out_features = attention_dim, bias = False) 
        self.query_layer = nn.Linear(in_features = embedding_dim, out_features = attention_dim, bias = False) 
        self.value_layer = nn.Linear(in_features = embedding_dim, out_features = attention_dim, bias = False) 

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        query, key, value = self.query_layer(embedded), self.key_layer(embedded), self.value_layer(embedded) # batch_size x context_len x attention_dim
        attention_score = query @ torch.transpose(key, 1, 2) / math.sqrt(self.attention_dim) # batch_size x context_len x context_len

        batch_size, context_len, embedding_dim = embedded.shape
        mask = torch.tril(torch.ones(batch_size, context_len, context_len)) == 0
        attention_score = attention_score.masked_fill(mask, float('-inf')).softmax(dim = 2) # batch_size x context_len x context_len
        
        return torch.round(attention_score @ value, decimals = 4)