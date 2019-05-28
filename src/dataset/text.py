import itertools
from typing import Tuple, Any

from torch.utils.data import Dataset

from preprocessors.corpus import CorpusPreprocessor


class TextDataset(Dataset):
    def __init__(self, path: str, mask: str, seed: Any = None, limit: int = 0):
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # Create preprocessor and transform lines so all the words get cached
            preprocessor = CorpusPreprocessor(mask, seed)
            lines = [preprocessor.transform_text(line) for line in lines]
            # Finally transform lines into dataset
            self.sentences = [preprocessor.mask_text(line) for line in lines]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, item: int) -> Tuple[str, str, int]:
        return self.sentences[item]
