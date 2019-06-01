import torch
from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.pool = nn.MaxPool1d(pool_size)
        self.conv1 = nn.Conv1d(alignment, embedding_size * 2, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(embedding_size * 2, embedding_size * 4, kernel_size=kernel_size)
        n = alphabet_size - 8
        m = embedding_size * 2 // pool_size
        self.fc = nn.Linear(n * m, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
