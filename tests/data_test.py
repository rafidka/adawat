# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import pytest
import torch
from torch.utils.data import DataLoader
from adawat.data import ListBasedDataset, ZipDataset


class TestListBasedDataset:
    features = list(range(10))
    targets = list(range(10))
    letter_features = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
    letter_targets = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']

    def test_getitem(self):
        dataset = ListBasedDataset(self.features, self.targets)
        for i, (X, y) in enumerate(dataset):
            assert (X, y) == (self.features[i], self.targets[i])

    def test_getitem_with_transform(self):
        def transform_feature(feature):
            return 3 * feature

        def transform_target(target):
            return 2 * target

        dataset = ListBasedDataset(self.features, self.targets,
                                   transform_feature, transform_target)
        for i, (X, y) in enumerate(dataset):
            X_transformed = transform_feature(self.features[i])
            y_transformed = transform_target(self.targets[i])
            assert (X, y) == (X_transformed, y_transformed)

    def test_len(self):
        dataset = ListBasedDataset(self.features, self.targets)
        assert len(self.features) == len(dataset)

    def test_use_dataloader(self):
        dataset = ListBasedDataset(self.features, self.targets)
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (X, y) in enumerate(dataloader):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            X_org = torch.tensor(self.features[batch_start:batch_end])
            y_org = torch.tensor(self.targets[batch_start:batch_end])
            assert torch.all(torch.eq(X, X_org)) == True
            assert torch.all(torch.eq(y, y_org)) == True

    def test_use_dataloader_with_letters(self):
        dataset = ListBasedDataset(self.letter_features, self.letter_targets)
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (X, y) in enumerate(dataloader):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            assert X == tuple(self.letter_features[batch_start:batch_end])
            assert y == tuple(self.letter_targets[batch_start:batch_end])


class TestZipDataset:
    list1 = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
    list2 = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']

    def test_iterate(self):
        dataset = ZipDataset(self.list1, self.list2)
        for i, (item1, item2) in enumerate(dataset):
            assert (item1, item2) == (self.list1[i], self.list2[i])

    def test_iterate_transform(self):
        dataset = ZipDataset(self.list1, self.list2,
                             transform=lambda l1, l2: l1 + l2)
        for i, item in enumerate(dataset):
            assert item == self.list1[i] + self.list2[i]

    def test_len(self):
        dataset = ZipDataset(self.list1, self.list2, length=len(self.list1))
        assert len(self.list1) == len(dataset)

    def test_len_not_specified(self):
        dataset = ZipDataset(self.list1, self.list2)
        with pytest.raises(AttributeError):
            len(dataset)
