import itertools
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from preprocessor import CorpusPreprocessor
from utils.text import TensorEncoder


class CorpusDataset(Dataset):
    def __init__(self, path: str, preprocessor: CorpusPreprocessor, encoder: TensorEncoder, limit: int = 0):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # First transform all the lines to cache words in corpus
            lines = [preprocessor.transform_text(line) for line in lines]
            # Transform lines into test format
            for line in lines:
                sentence, masked, valid = preprocessor.mask_text(line)
                # Transform text into tensors
                sample = encoder.encode_sentence('{} # {}'.format(sentence, masked))
                self.samples.append((sample, valid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        return self.samples[item]
