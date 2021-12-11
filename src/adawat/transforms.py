from typing import Any, TypeVar, Generic, Callable, List

import numpy as np
import torch


class Compose(object):
    """
    Compose multiple transforms together, applying the first, then the second,
    and so on until the last.
    """

    def __init__(self, *args):
        self.transforms = args

    def __call__(self, obj):
        for transform in self.transforms:
            obj = transform(obj)
        return obj


TWord = TypeVar('TWord')
TIdx = TypeVar('TIdx')


class WordToIndex(Generic[TWord, TIdx]):
    """
    Converts words to indices based on a dictionary.
    """

    def __init__(self, word2idx: Callable[[TWord], TIdx]):
        """
        word2idx -- A function that retrieves a unique index for a word.
        """
        self.word2idx = word2idx

    def __call__(self, word: TWord) -> TIdx:
        return self.word2idx(word)


class WordsToIndices(Generic[TWord, TIdx]):
    """
    List-version of WordToIndex transform, i.e. coverts multiple words to their
    corresponding indices.
    """

    def __init__(self, word2idx: Callable[[TWord], TIdx]):
        """
        word2idx -- A function that retrieves a unique index for a word.
        """
        self.word2idx = word2idx

    def __call__(self, words: List[TWord]) -> List[TIdx]:
        return [self.word2idx(word) for word in words]


class WordToOneHot(Generic[TWord, TIdx]):
    """
    Converts words to one-hot encoding based on a dictionary.
    """

    def __init__(self, word2idx: Callable[[TWord], TIdx], vocab_size: int):
        """
        word2idx -- A function that retrieves a unique index for a word.
        vocab_size -- The size of the dictionary. This is used to decide the
                      size of the one-hot vector.
        """
        self.word2idx = word2idx
        self.vocab_size = vocab_size

    def __call__(self, word: TWord) -> TIdx:
        idx = self.word2idx(word)
        onehot = [0] * self.vocab_size
        onehot[idx] = 1
        return onehot


class ToPyTorchTensor(object):
    """
    Converts a Python list or a NumPy list to a PyTorch vector.
    """

    def __init__(self, dtype, device=None):
        self.dtype = dtype
        self.device = device

    def __call__(self, list):
        if self.device is not None:
            return torch.tensor(list, dtype=self.dtype, device=self.device)
        else:
            return torch.tensor(list, dtype=self.dtype)
