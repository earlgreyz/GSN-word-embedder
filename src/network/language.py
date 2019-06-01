import torch
from torch import nn

from network.embedder import Embedder


class Language(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int):
        super().__init__()
        self.embedder = Embedder(alignment, alphabet_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, W, L = x.shape
        x = x.reshape(-1, W, L)
        x = self.embedder(x)
        x = x.reshape(N, M, -1)
        return x
