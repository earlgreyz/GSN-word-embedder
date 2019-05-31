from torch import nn


class Language(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

    def forward(self, *input):
        pass