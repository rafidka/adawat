import torch
from torch import nn
from typing import Any, Callable, List


class ListDataset(torch.utils.data.Dataset):
    """Converts a normal Python list into a PyTorch Dataset for use with
    DataLoaders."""

    def __init__(self, features: List, targets: List, transform: Callable[[Any], torch.tensor] = None):
        if len(features) != len(targets):
            raise ValueError(f"X and y must be of the same length. Received " +
                             "{len(X)} for X and {len(y)} for y.")
        self.features = features
        self.targets = targets
        # save length in case it is expensive to calculate
        self.len = len(self.features)
        self.transform = transform if transform is not None else lambda x: x

    def __getitem__(self, idx: int) -> Any:
        return self.transform((self.features[idx], self.targets[idx]))

    def __len__(self) -> int:
        return self.len
