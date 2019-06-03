import random
import re
from typing import Any, Tuple


def cache_corpus(func):
    """
    Wraps a method causing it to store its results in the `corpus` field.
    Each time a function is called words from the returned sentence are
    added to the `corpus` member field.
    :param func: function to wrap
    :return: the result of the `func` function
    """

    def wrapped(self, *args, **kwargs) -> str:
        text = func(self, *args, **kwargs)
        self.corpus.update(text.split())
        return text

    return wrapped


class CorpusPreprocessor:
    def __init__(self, mask: str, seed: Any = None):
        self.mask = mask
        self.rand = random.Random(seed)
        self.corpus = set()

    @cache_corpus
    def transform_text(self, text: str) -> str:
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-ZąĄćĆęĘłŁńŃóÓśŚżŻźŹ ]', '', text)
        # Replace consecutive spaces with a single space and strip
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def mask_text(self, text: str) -> Tuple[str, str, int]:
        words = text.split(' ')
        # Mask a random word in the sentence
        i = self.rand.randrange(len(words))
        word = words[i]
        words[i] = self.mask
        # Pick and return original or random word
        p = self.rand.randint(0, 1)
        if p == 0:
            word = self.random_word()
        return ' '.join(words), word, p

    def random_word(self) -> str:
        return self.rand.sample(self.corpus, 1)[0]
