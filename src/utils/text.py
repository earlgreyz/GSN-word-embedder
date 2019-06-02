import numpy as np
import torch

ALPHABET_PL = 'aAąĄbBcCćĆdDeEęĘfFgGhHiIjJkKlLłŁmMnNńŃoOóÓpPqQrRsSśŚtTuUvVwWxXyYzZżŻźŹ?#'


class Encoder:
    def __init__(self, alignment: int, alphabet: str = ALPHABET_PL):
        self.char_align = alignment
        self.N = len(alphabet)
        self.int2char = dict(enumerate(alphabet))
        self.char2int = {char: i for i, char in self.int2char.items()}

    def encode_word(self, text: str) -> torch.Tensor:
        if self.char_align < len(text):
            raise RuntimeError('Encoded word cannot be longer than char alignment')
        encoding = np.zeros((self.char_align, self.N), dtype=np.float32)
        for i, c in enumerate(text):
            encoding[i, self.char2int[c]] = 1
        return torch.from_numpy(encoding)

    def encode_sentence(self, text: str) -> torch.Tensor:
        words = text.split()
        encoding = np.zeros((len(words), self.char_align, self.N), dtype=np.float32)
        for i, word in enumerate(words):
            encoding[i, :, :] = self.encode_word(word)
        return torch.from_numpy(encoding)
