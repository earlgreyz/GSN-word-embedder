import torch
from torch import nn

from network.embedder import Embedder


class Language(nn.Module):
    def __init__(self, alignment: int, alphabet_size: int, embedding_size: int, hidden_size: int = 8):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedder = Embedder(alignment, alphabet_size, embedding_size)
        self.drouput = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        N, M, W, L = x.shape
        # Reshape such that embedder gets a batch of encoded words
        x = x.reshape(N * M, W, L)
        x = self.embedder(x)
        x = self.drouput(x)
        # Reshape to the form of (batch, sentence, word)
        x = x.reshape(N, M, self.embedding_size)
        # Run LSTM on the sentences
        x, _ = self.lstm(x)
        out = torch.zeros((N, self.hidden_size * 2), device=x.device)
        for i in range(N):
            out[i, :] = x[i, l[i], :].squeeze()
        # Fully connected layer to output classes
        out = self.fc(out)
        return out
