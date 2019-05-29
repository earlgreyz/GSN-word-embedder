import itertools
from typing import Any, List, Set

from preprocessors.corpus import CorpusPreprocessor


class Corpus:
    def __init__(self, path: str, mask: str, seed: Any = None, limit: int = 0):
        self.preprocessor = CorpusPreprocessor(mask, seed)
        with open(path) as f:
            # Load only portion of dataset if limit is specified
            lines = itertools.islice(f, limit) if limit > 0 else f
            # Cache for the corpus contents
            self.sentences: List[str] = []
            words: Set[str] = set()
            # Transform the lines and cache the words in the corpus
            for line in lines:
                sentence = self.preprocessor.transform_text(line)
                words.update(sentence.split(' '))
                self.sentences.append(sentence)
            # Update the preprocessor with the words in the corpus
            self.preprocessor.words = list(words)

    @property
    def words(self) -> List[str]:
        return self.preprocessor.words
