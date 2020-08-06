import os
from itertools import islice
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
    def __init__(self, filepath: str, max_lines=None):
        self._load_raw_text(filepath, max_lines)
        self._extract_tokens()
        self._build_vocab()
        self._build_idxs()

    def _load_raw_text(self, filepath: str, max_lines=None):
        with open(filepath, "r") as f:
            if max_lines is not None:
                self._raw_text = os.linesep.join(
                    islice(f.readlines(), max_lines))
            else:
                self._raw_text = f.read()

    def _extract_tokens(self):
        self.tokens = nltk.word_tokenize(self._raw_text)

    def _build_vocab(self):
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
