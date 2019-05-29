import random
import re
from typing import Any, Tuple, List


class CorpusPreprocessor:
    def __init__(self, mask: str, seed: Any = None):
        self.mask = mask
        self.rand = random.Random(seed)
        self.words: List[str] = []

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
        return self.rand.choice(self.words)
