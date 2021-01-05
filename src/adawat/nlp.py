import os
from collections import Counter
from itertools import islice
from typing import Callable, Iterable
import gzip
import nltk
from adawat.serialization import stateful


UNKNOWN_WORD = '<unk>'


# TODO Disabling stateful-ness at this point, since the lines argument, which
# is an iterator, might have different values even for the same content, hence
# resulting in a new state ID every time. In fact, even if it is not an
# iterator, e.g. list, we probably don't want to use its string representation
# as part of the ID. We should probably allow the user to specify an ID or modify
# the @stateful decorator to only use the ID as the key of the object.
# 1. Modifying
# @stateful(attrs=[
#     'tokens_per_line',
#     'tokens',
#     'vocab',
#     'vocab_size',
#     '_word2idx',
#     '_idx2word',
# ])
class Corpus():
    def __init__(self,
                 lines: Iterable[str],
                 max_vocab: int = None,
                 min_word_freq: int = None,
                 preprocessor: Callable[[str], str] = None):
        self._extract_tokens(lines, preprocessor)
        self._build_vocab(max_vocab, min_word_freq)
        self._build_idxs()

    def _extract_tokens(self,
                        lines: Iterable[str],
                        preprocessor: Callable[[str], str] = None):
        if preprocessor is None:
            def preprocessor(line: str) -> str: return line
        self.tokens_per_line = [nltk.word_tokenize(preprocessor(line))
                                for line in lines]
        self.tokens = [token
                       for line_tokens in self.tokens_per_line
                       for token in line_tokens]

    def _build_vocab(self, max_vocab=None, min_word_freq=None):
        counter = Counter(self.tokens).most_common(max_vocab)
        if min_word_freq is None:
            min_word_freq = 1
        self.vocab = [UNKNOWN_WORD] + [key
                                       for key, count in counter
                                       if count >= min_word_freq]
        self.vocab_size = len(self.vocab)

    def _build_idxs(self):
        self._word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self._idx2word = {idx: word for idx, word in enumerate(self.vocab)}

    def word2idx(self, word: str) -> int:
        # If no matching word is found, return the index for the special <unk> word.
        return self._word2idx.get(word, self._word2idx[UNKNOWN_WORD])

    def idx2word(self, idx: int) -> str:
        return self._idx2word[idx]
