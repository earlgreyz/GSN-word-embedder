import itertools
from typing import Tuple

import torch
from torch.utils.data import Dataset

from preprocessor import CorpusPreprocessor
from utils import text


class CorpusDataset(Dataset):
    def __init__(self, path: str, preprocessor: CorpusPreprocessor, encoder: text.Encoder, limit: int = 0):
        self.samples = []
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # First transform all the lines to cache words in corpus
            lines = [preprocessor.transform_text(line) for line in lines]
            # Transform lines into test format
            for line in lines:
                sentence, masked, valid = preprocessor.mask_text(line)
                sample = encoder.encode_sentence(' '.join([sentence, '#', masked]))
                self.samples.append((sample, torch.tensor(valid)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[item]
