import sys
import os
import torch.nn as nn

sys.path.append(os.path.abspath("../src"))
from LayerNorm import LayerNorm
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.drop_rate = config["drop_rate"]
        self.qkv_bias = config["qkv_bias"]
        self.out_bias = config["out_bias"]

        self.layerNorm1 = LayerNorm(self.emb_dim)
        self.mha = MultiHeadAttention(self.emb_dim, self.emb_dim, self.n_heads, self.drop_rate, self.qkv_bias, self.out_bias)
        self.dropout1 = nn.Dropout(self.drop_rate)
        
        self.layerNorm2 = LayerNorm(self.emb_dim)
        self.feedForward = FeedForward(self.emb_dim)
        self.dropout2 = nn.Dropout(self.drop_rate)

    def forward(self, x):
        # First residual block: Multi-Head Attention
        shortcut = x
        x = self.layerNorm1(x)
        x = self.mha(x)
        x = self.dropout1(x)
        x = x + shortcut  # Residual connection
        
        # Second residual block: Feed Forward
        shortcut = x
        x = self.layerNorm2(x)
        x = self.feedForward(x)
        x = self.dropout2(x)
        x = x + shortcut  # Residual connection
        
        return x
    