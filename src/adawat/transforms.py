from typing import TypeVar, Generic, Callable, List
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


TWord = TypeVar('T')
TIdx = TypeVar('T')


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
    List-version of WordToIndex transform, i.e. coverts multiple worsd to their
    corresponding indices.
    """

    def __init__(self, word2idx: Callable[[TWord], TIdx]):
        """
        word2idx -- A function that retrieves a unique index for a word.
        """
        self.word2idx = word2idx

    def __call__(self, words: List[TWord]) -> List[TIdx]:
        return [self.word2idx(word) for word in words]


class ToPyTorchTensor(object):
    """
    Converts a Python list or a NumPy list to a PyTorch vector.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, list):
        return torch.tensor(list, dtype=self.dtype)
