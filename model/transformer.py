import torch
import torch.nn as nn
from .positional import PositionalEmbedding
from .encoder import Encoder
from .patch_embed import PatchEmbedding


class Transformer(nn.Module):
    def __init__(self, classes, d_model, heads, d_ff,
                 num_encoder_layers, patches, img_size, 
                 patch_size, in_channels, dropout=0.1):
        super().__init__()

        self.pos = PositionalEmbedding(patches+1, d_model)
        self.encoder = Encoder(d_model, heads, d_ff, num_encoder_layers, dropout)
        self.output_proj = nn.Linear(d_model, classes)
        self.patch_emb = PatchEmbedding(img_size, patch_size, in_channels, d_model)

    def encode(self, src):
        x = self.patch_emb(src)
        x = self.pos(x)
        return self.encoder(x)

    def forward(self, src):
        # src: image batch [batch, in_channels, img_size, img_size].
        # Returns class logits [batch, classes] from the CLS token.
        out = self.encode(src)
        return self.output_proj(out[:, 0])
