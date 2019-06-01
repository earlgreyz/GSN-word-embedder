import torch
from torch import nn

from network.embedder import Embedder


class Language(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = Embedder(alignment, alphabet_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        N, M, W, L = x.shape
        x = x.reshape(-1, W, L)
        x = self.embedder(x)
        x = x.reshape(N, M, -1)

        hidden = torch.zeros(2, N, self.hidden_size)
        out, hidden = self.rnn(x, hidden)
        #out = out.contiguous().view(-1, out.shape[2])
        out = out[:, M - 1, :]
        out = self.fc(out)
        return out, hidden
