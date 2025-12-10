import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("../src"))
from LayerNorm import LayerNorm
from TransformerBlock import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.drop_rate = config["drop_rate"]
        self.qkv_bias = config["qkv_bias"]

        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim) # 50257 * 768 = 38,605,056
        self.pos_embedding_layer = nn.Embedding(self.context_length, self.emb_dim) # 1024 * 768 = 786,432

        self.dropout = nn.Dropout(self.drop_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layers)]) 
        # LayerNorm1 -> 2 * 768 = 1,536
        # qkv + out -> (3 * (768 * 768)) + (768 * 768) = 2,359,296
        # 2,359,296 * 12 = 28,311,552
        # LayerNorm2 -> 2 * 768 = 1,536
        # FeedForward -> 768 * (4 * 768) + (4 * 768) * 768 = 4,718,592

        self.layerNorm = LayerNorm(self.emb_dim) # 2 * 768 = 1,536

        self.inverseEmbedding = nn.Linear(self.emb_dim, self.vocab_size, bias=False) # 768 * 50257 = 38,605,056

    def forward(self, x):
        # x: (batch_size, context_length)
        batch_size, seq_len = x.shape

        token_embeddings = self.token_embedding(x)

        pos_embeddings = self.pos_embedding_layer(torch.arange(seq_len, device=x.device))  # (seq_len, emb_dim)
        pos_embeddings = pos_embeddings.unsqueeze(0)  # (1, seq_len, emb_dim)

        x = token_embeddings + pos_embeddings
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layerNorm(x)

        logits = self.inverseEmbedding(x)
        
        return logits
    