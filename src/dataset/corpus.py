import itertools
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from preprocessor import CorpusPreprocessor
from utils.text import TensorEncoder


class CorpusDataset(Dataset):
    def __init__(self, path: str, preprocessor: CorpusPreprocessor, encoder: TensorEncoder, limit: int = 0):
        self.samples: List[Tuple[List[torch.Tensor], int]] = []
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # First transform all the lines to cache words in corpus
            lines = [preprocessor.transform_text(line) for line in lines]
            # Transform lines into test format
            for line in lines:
                sentence, masked, valid = preprocessor.mask_text(line)
                # Transform text into tensors
                words = [encoder.encode(word) for word in sentence.split()]
                words.append(encoder.encode(masked))
                self.samples.append((words, valid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[List[torch.Tensor], int]:
        return self.samples[item]
