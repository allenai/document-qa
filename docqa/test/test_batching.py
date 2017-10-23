import unittest

import numpy as np

from docqa.dataset import ClusteredBatcher, ShuffledBatcher


class TestBatches(unittest.TestCase):

    def assert_unique(self, batches):
        values, c = np.unique(np.concatenate(batches), return_counts=True)
        self.assertTrue(np.all(c == 1))

    def test_unique_samples(self):
        batchers = [ShuffledBatcher(5), ClusteredBatcher(5, lambda x:x)]
        test_u = self.assert_unique
        for batcher in batchers:
            batches = list(batcher.get_epoch(list(np.arange(21))))
            self.assertEqual(len(batches), 4)
            self.assertEqual(batcher.epoch_size(21), 4)
            test_u(batches)

            batches = list(batcher.get_epoch(list(np.arange(25))))
            self.assertEqual(len(batches), 5)
            self.assertEqual(batcher.epoch_size(25), 5)
            test_u(batches)

            batches = list(batcher.get_epoch(list(np.arange(5))))
            self.assertEqual(len(batches), 1)
            self.assertEqual(batcher.epoch_size(5), 1)
            test_u(batches)

    def test_truncate_samples(self):
        batchers = [ShuffledBatcher(5, truncate_batches=True), ClusteredBatcher(5, lambda x: x, truncate_batches=True)]
        test_u = self.assert_unique
        for batcher in batchers:
            batches = list(batcher.get_epoch(list(np.arange(21))))
            self.assertEqual(len(batches), 5)
            self.assertEqual(batcher.epoch_size(21), 5)
            test_u(batches)

            batches = list(batcher.get_epoch(list(np.arange(4))))
            self.assertEqual(len(batches), 1)
            self.assertEqual(batcher.epoch_size(4), 1)
            test_u(batches)

            batches = list(batcher.get_epoch(list(np.arange(10))))
            self.assertEqual(len(batches), 2)
            self.assertEqual(batcher.epoch_size(10), 2)
            test_u(batches)

    def test_order(self):
        batch = list(np.arange(103))
        np.random.shuffle(batch)
        batches = list(ClusteredBatcher(10, lambda x: x, truncate_batches=True).get_epoch(batch))
        self.assertEqual(len(batches), 11)
        for batch in batches:
            for i in range(0, len(batch)-1):
                if batch[i] != batch[i+1]-1:
                    raise ValueError("Out of order point")

