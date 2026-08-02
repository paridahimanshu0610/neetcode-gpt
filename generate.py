import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        context_so_far = context
        res = []

        for i in range(new_chars):
            if context.shape[1] > context_length:
                context_so_far = context_so_far[:, -context_length:] 

            y_pred_logits = model(context_so_far)

            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            predicted_token = torch.multinomial(torch.nn.Softmax(dim = 1)(y_pred_logits[:, -1, :]), num_samples=1, generator=generator)

            context_so_far = torch.cat((context_so_far, predicted_token), dim = 1)
            res.append(int_to_char[predicted_token.item()])

        return "".join(res)