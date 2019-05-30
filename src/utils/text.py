from typing import Dict

import numpy as np
import torch

ALPHABET_PL = 'aAąĄbBcCćĆdDeEęĘfFgGhHiIjJkKlLłŁmMnNńŃoOóÓpPqQrRsSśŚtTuUvVwWxXyYzZżŻźŹ?'


class TensorEncoder:
    def __init__(self, encoding_length: int, alphabet: str = ALPHABET_PL):
        self.encoding_length = encoding_length
        self.alphabet_length = len(alphabet)
        self.size = (self.encoding_length, self.alphabet_length)
        self.int2char: Dict[int, str] = dict(enumerate(alphabet))
        self.char2int: Dict[str, int] = {char: i for i, char in self.int2char.items()}

    def encode(self, text: str) -> torch.Tensor:
        if self.encoding_length < len(text):
            raise RuntimeError('Encoded text cannot be longer than its encoding')
        encoding = np.zeros(self.size, dtype=np.float32)
        for i, c in enumerate(text):
            encoding[i, self.char2int[c]] = 1
        return torch.from_numpy(encoding)
