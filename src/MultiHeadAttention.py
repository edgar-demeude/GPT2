import torch
import torch.nn as nn
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, d_out=2, num_head=2, dropout=0.0):
        super().__init__()
        assert d_out % num_head == 0 # d_out must be divisible by num_head
        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out // num_head
        self.attn_dropout = torch.nn.Dropout(dropout)

        # linear projections for query, key and value
        self.W_query = torch.nn.Linear(input_dim, d_out, bias=False)
        self.W_key = torch.nn.Linear(input_dim, d_out, bias=False)
        self.W_value = torch.nn.Linear(input_dim, d_out, bias=False)

        # output projection to combine heads
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=False)

    def forward(self, x):
        # x: (T, D) or (B, T, D) -> returns (T, d_out) or (B, T, d_out)
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True

        B, T, _ = x.shape

        # project to Q, K, V -> (B, T, d_out)
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # reshape to separate heads: (B, T, num_head, head_dim) -> (B, num_head, T, head_dim)
        q = q.view(B, T, self.num_head, self.head_dim).transpose(1,2)  # (B, num_head, T, head_dim)
        k = k.view(B, T, self.num_head, self.head_dim).transpose(1,2)  # (B, num_head, T, head_dim)
        v = v.view(B, T, self.num_head, self.head_dim).transpose(1,2)  # (B, num_head, T, head_dim)

        # attention scores: use k.transpose to get (B, num_head, head_dim, T), result (B, num_head, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # attention weights
        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        # combine attention with values -> (B, num_head, T, head_dim)
        out = torch.matmul(attention, v)

        # transpose and merge heads -> (B, T, d_out)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)

        # final linear projection
        out = self.out_proj(out)

        if squeezed:
            return out.squeeze(0)
        return out