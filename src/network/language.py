from torch import nn


class Language(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, bidirectional=True)

    def forward(self, *input):
        pass