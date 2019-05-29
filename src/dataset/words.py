from torch.utils.data import Dataset

from dataset.corpus import Corpus


class WordsDataset(Dataset):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.corpus.words)

    def __getitem__(self, item: int) -> str:
        return self.corpus.words[item]
