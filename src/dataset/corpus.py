import itertools
from typing import Any, Tuple

from torch.utils.data import Dataset

from preprocessor import CorpusPreprocessor


class CorpusDataset(Dataset):
    def __init__(self, path: str, mask: str, seed: Any = None, limit: int = 0):
        preprocessor = CorpusPreprocessor(mask, seed)
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # First transform all the lines to cache words in corpus
            lines = [preprocessor.transform_text(line) for line in lines]
            # Transform lines into test format
            self.samples = [preprocessor.mask_text(line) for line in lines]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[str, str, int]:
        return self.samples[item]
