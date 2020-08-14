import os
from collections import Counter
from itertools import islice
from typing import Callable
import nltk
from adawat.serialization import stateful


UNKNOWN_WORD = '<unk>'


@stateful(attrs=[
    'tokens',
    'vocab',
    'vocab_size',
    '_word2idx',
    '_idx2word',
])
class Corpus():
    def __init__(self, filepath: str, max_lines=None, max_vocab=None,
                 preprocessor: Callable[[str], str] = None):
        self._load_raw_text(filepath, max_lines, preprocessor)
        self._extract_tokens()
        self._build_vocab(max_vocab)
        self._build_idxs()

    def _load_raw_text(self, filepath: str, max_lines=None,
                       preprocessor: Callable[[str], str] = None):
        if preprocessor is None:
            def preprocessor(line): return line

        with open(filepath, "r") as f:
            if max_lines is not None:
                self._raw_text = [preprocessor(line)
                                  for line in islice(f, max_lines)]
            else:
                self._raw_text = [preprocessor(line)
                                  for line in f]

    def _extract_tokens(self):
        self.tokens_per_line = [nltk.word_tokenize(line)
                                for line in self._raw_text]
        self.tokens = [token
                       for line_tokens in self.tokens_per_line
                       for token in line_tokens]

    def _build_vocab(self, max_vocab=None):
        if max_vocab is not None:
            self.vocab = [UNKNOWN_WORD] + [key for key,
                                           value in Counter(self.tokens).most_common(max_vocab)]
        else:
            self.vocab = [UNKNOWN_WORD] + list(set(self.tokens))
        self.vocab_size = len(self.vocab)

    def _build_idxs(self):
        self._word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self._idx2word = {idx: word for idx, word in enumerate(self.vocab)}

    def word2idx(self, word: str) -> int:
        # If no matching word is found, return the index for the special <unk> word.
        return self._word2idx.get(word, self._word2idx[UNKNOWN_WORD])

    def idx2word(self, idx: int) -> str:
        return self._idx2word[idx]
