import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.heads = heads
        self.d_model = d_model
        self.head_dim = d_model // heads

        # Fused per-head projections in a single matmul, split out by reshaping.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value):
        # query/key/value: [batch, seq, d_model] (same tensor for self-attention).
        batch_size = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        A = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(A, V)  # [batch, heads, seq, head_dim]

        # Concatenate heads back to [batch, seq, d_model].
        x = x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_model)

        return self.W_o(x)
