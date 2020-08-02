# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest
import torch
from torch.utils.data import DataLoader
from adawat.data import ListDataset


class TestListDataset(unittest.TestCase):
    features = list(range(10))
    targets = list(range(10))
    letter_features = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1']
    letter_targets = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']

    def test_getitem(self):
        dataset = ListDataset(self.features, self.targets)
        for i, (X, y) in enumerate(dataset):
            self.assertEqual((X, y), (self.features[i], self.targets[i]))

    def test_getitem_with_transform(self):
        def transform(X_y):
            X, y = X_y
            return (3 * X, 2 * y)

        dataset = ListDataset(self.features, self.targets, transform)
        for i, (X, y) in enumerate(dataset):
            self.assertEqual((X, y), transform((
                self.features[i], self.targets[i])))

    def test_len(self):
        dataset = ListDataset(self.features, self.targets)
        self.assertEqual(len(self.features), len(dataset))

    def test_use_dataloader(self):
        dataset = ListDataset(self.features, self.targets)
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (X, y) in enumerate(dataloader):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            X_org = torch.tensor(self.features[batch_start:batch_end])
            y_org = torch.tensor(self.targets[batch_start:batch_end])
            self.assertTrue(torch.all(torch.eq(X, X_org)))
            self.assertTrue(torch.all(torch.eq(y, y_org)))

    def test_use_dataloader_with_letters(self):
        dataset = ListDataset(self.letter_features, self.letter_targets)
        batch_size = 4
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (X, y) in enumerate(dataloader):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            self.assertEqual(
                X, tuple(self.letter_features[batch_start:batch_end]))
            self.assertEqual(
                y, tuple(self.letter_targets[batch_start:batch_end]))
