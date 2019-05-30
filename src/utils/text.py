from typing import Dict

import numpy as np
import torch

ALPHABET_PL = 'aAąĄbBcCćĆdDeEęĘfFgGhHiIjJkKlLłŁmMnNńŃoOóÓpPqQrRsSśŚtTuUvVwWxXyYzZżŻźŹ?#'


class TensorEncoder:
    def __init__(self, char_align: int, word_align: int, alphabet: str = ALPHABET_PL):
        self.char_align = char_align
        self.word_align = word_align
        self.N = len(alphabet)
        self.int2char: Dict[int, str] = dict(enumerate(alphabet))
        self.char2int: Dict[str, int] = {char: i for i, char in self.int2char.items()}

    def encode_word(self, word: str) -> torch.Tensor:
        if self.char_align < len(word):
            raise RuntimeError('Encoded word cannot be longer than char alignment')
        encoding = np.zeros((self.char_align, self.N), dtype=np.float32)
        for i, c in enumerate(word):
            encoding[i, self.char2int[c]] = 1
        return torch.from_numpy(encoding)

    def encode_sentence(self, text: str) -> torch.Tensor:
        words = text.split()
        if self.word_align < len(words):
            raise RuntimeError('Encoded text cannot have more words than word alignment')
        encoding = np.zeros((self.word_align, self.char_align, self.N), dtype=np.float32)
        for i, word in enumerate(words):
            encoding[i, :, :] = self.encode_word(word)
        return torch.from_numpy(encoding)
