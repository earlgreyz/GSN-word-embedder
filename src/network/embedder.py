from typing import Union

from torch import nn
import torch.nn.functional as F

from utils import text


class Embedder(nn.Module):
    def __init__(self, embedding_size: int, encoder: text.Encoder,
                 out: int = 32, kernel: Union[int, tuple] = 3, pool: int = 2):
        super().__init__()
        self.encoder = encoder
        self.convolution = nn.Conv2d(1, out, kernel, padding=1)
        self.pool = nn.MaxPool2d(pool)
        self.full = nn.Linear(out, embedding_size)

    def forward(self, x):
        x = self.pool(F.relu(self.convolution(x)))
        return self.full(x)
