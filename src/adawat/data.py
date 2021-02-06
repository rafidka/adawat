import torch
from torch import nn
from typing import Any, Callable, List, Iterable


class ListDataset(torch.utils.data.Dataset):
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


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, features: Iterable, targets: Iterable,
                 features_transform: Callable[[Any], torch.tensor] = None,
                 targets_transform: Callable[[Any], torch.tensor] = None,
                 length: int = None):
        self.features = features
        self.targets = targets
        self.features_transform = features_transform if features_transform is not None else lambda x: x
        self.targets_transform = targets_transform if targets_transform is not None else lambda x: x
        self.length = length

    def __len__(self):
        if self.length is None:
            raise RuntimeError(
                "Length was not specified during the instantiation of this " +
                "dataset. For instances of IterableDataset, the length " +
                "cannot be found without iterating through the iterable " +
                "till the end, which could be an expensive operation and " +
                "is against the nature of iterables.")
        return self.length

    def __iter__(self):
        def transform(feature, target):
            return (self.features_transform(feature), self.targets_transform(target))

        return map(transform, self.features, self.targets)
