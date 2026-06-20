import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding added to the token sequence."""

    def __init__(self, patches, d_model):
        super().__init__()
        self.pos_weights = nn.Parameter(torch.zeros(patches, d_model))
        nn.init.trunc_normal_(self.pos_weights, std=0.02)

    def forward(self, x):
        # x: (batch, patches, d_model)
        return x + self.pos_weights