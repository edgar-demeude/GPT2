import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.layer1 = nn.Linear(emb_size, 4*emb_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(4*emb_size, emb_size)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))