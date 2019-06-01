import torch
from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(alignment, embedding_size, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(pool_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        n = (alphabet_size - 1) // pool_size
        m = embedding_size // pool_size
        self.fc = nn.Linear(n * m, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.size(0)
        x = self.pool(self.bn(F.relu(self.conv(x))))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
