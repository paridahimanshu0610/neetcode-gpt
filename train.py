import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

        model.train()
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            positions = torch.randint(0, data.shape[0]-context_length, (batch_size,))
            X = torch.stack([data[pos:pos+context_length] for pos in positions])
            y = torch.stack([data[pos+1:pos+1+context_length] for pos in positions])

            optimizer.zero_grad()
            logits = model(X)
            B, T, C = logits.shape # Shape: batch_size, context_length, vocab_size

            train_loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))

            train_loss.backward()
            optimizer.step()


        return round(train_loss.item(), 4)
