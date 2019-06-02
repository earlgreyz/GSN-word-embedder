import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int):
        super().__init__()
        self.fc = nn.Linear(alignment * alphabet_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
