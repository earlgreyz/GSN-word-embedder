import click

import itertools

import torch
from torch.utils.data import Dataset

from preprocessor import CorpusPreprocessor
from utils import text


class WordsDataset(Dataset):
    def __init__(self, path: str, encoder: text.Encoder, limit: int = 0):
        self.encodings = []

        with open(path) as f:
            # Create preprocessor
            preprocessor = CorpusPreprocessor(mask='')
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # First transform all the lines to cache words in corpus
            with click.progressbar(lines, label='Transforming sentences') as bar:
                for line in bar:
                    preprocessor.transform_text(line)

            self.words = list(preprocessor.corpus)
            # One-hot encode all words
            with click.progressbar(self.words, label='Encoding words') as bar:
                for word in bar:
                    self.encodings.append(encoder.encode_word(word))

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.encodings[item]
