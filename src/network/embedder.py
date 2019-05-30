from typing import Union

from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, embedding_size: int, out: int = 32, kernel: Union[int, tuple] = 3, pool: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(1, out, kernel)
        self.pool = nn.MaxPool2d(pool)
        self.full = nn.Linear(out, embedding_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.full(x)
