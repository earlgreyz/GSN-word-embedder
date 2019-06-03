import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int):
        super().__init__()
        self.lstm = nn.LSTM(alphabet_size, hidden_size=2, batch_first=True)
        self.fc = nn.Linear(alignment * 2, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x, _ = self.lstm(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        return x
