import torch
from torch import nn
from typing import Any, Callable, List, Iterable, TypeVar

T = TypeVar('T')
S = TypeVar('S')


# TODO Remove the 'targets' from this class; we don't need the dataset to be
# specific to features and targets tuples as we can do that by combining
# multiple datasets, e.g. via the ZipDataset.
class ListBasedDataset(torch.utils.data.Dataset):
    """Converts a normal Python list into a PyTorch Dataset for use with
    DataLoaders."""

    def __init__(self, features: List, targets: List,
                 features_transform: Callable[[Any], torch.tensor] = None,
                 targets_transform: Callable[[Any], torch.tensor] = None):
        if len(features) != len(targets):
            raise ValueError("X and y must be of the same length. Received " +
                             f"{len(features)} for X and {len(targets)} for y.")
        self.features = features
        self.targets = targets
        # save length in case it is expensive to calculate
        self.len = len(self.features)
        self.features_transform = features_transform if features_transform is not None else lambda x: x
        self.targets_transform = targets_transform if targets_transform is not None else lambda x: x

    def __getitem__(self, idx: int) -> Any:
        return (self.features_transform(self.features[idx]),
                self.targets_transform(self.targets[idx]))

    def __len__(self) -> int:
        return self.len


class IterableBasedDataset(torch.utils.data.IterableDataset):
    """
    A dataset based on a Python iterable.
    """

    def __init__(self,
                 iterable: Iterable,
                 transform: Callable = None,
                 length: int = None):
        """
        Constructs a new iterable-based dataset.

        Keyword arguments:
        iterable -- The iterable used in the dataset.
        transform -- (Optional) A transform to make some modification on each
            element before returning it.
        length -- (Optional) If specified, the iterable will have a length
            retrievable via the len() method.
        """
        self.iterable = iterable
        self.transform = transform if transform is not None else lambda x: x
        self.length = length

    def __len__(self):
        if self.length is None:
            # Raise AttributeError to mimic the behaviour of not having a
            # __len__ defined.
            raise AttributeError(
                "Length was not specified during the instantiation of this " +
                "dataset. For instances of IterableBasedDataset, the length " +
                "cannot be found without iterating through the iterable " +
                "till the end, which could be an expensive operation and " +
                "is against the nature of iterables.")
        return self.length

    def __iter__(self):
        return map(self.transform, self.iterable)


class ZipDataset(torch.utils.data.IterableDataset):
    def __init__(self, *args: Iterable, transform=None, length: int = None):
        self.iterables = args
        self.transform = transform
        self.length = length

    def __len__(self):
        if self.length is None:
            # Raise AttributeError to mimic the behaviour of not having a
            # __len__ defined.
            raise AttributeError(
                "Length was not specified during the instantiation of this " +
                "dataset. For instances of ZipDataset, the length " +
                "cannot be found without iterating through the iterable " +
                "till the end, which could be an expensive operation and " +
                "is against the nature of iterables.")
        return self.length

    def __iter__(self):
        if self.transform:
            return map(self.transform, *self.iterables)
        else:
            return zip(*self.iterables)
