import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Vecteur d'échelle entraînable
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Vecteur de décalage entraînable

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Moyenne par ligne (batch)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # Variance par ligne (batch)
        output = self.scale * ((x - mean) / (torch.sqrt(variance) + self.eps)) + self.shift
        return output