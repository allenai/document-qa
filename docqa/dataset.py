import itertools
from typing import Optional, Dict, Iterator, List, Callable

import numpy as np

from docqa.configurable import Configurable
from docqa.utils import ResourceLoader


class Dataset(object):
    """ Data iterator we can use to train or test a model on, responsible for both storing data
    and deciding how to batch it. """

    def get_epoch(self):
        """ Returns an iterator of batches/elements to train on, these elements are what will get
        passed to model.encode. Usually (but not necessarily) a list/batch of training examples """
        raise NotImplementedError(self.__class__)

    def get_batches(self, n_batches):
        if len(self) < n_batches:
            raise ValueError()
        return itertools.islice(self.get_epoch(), n_batches)

    def get_epochs(self, n_epochs: int):
        for _ in range(n_epochs):
            for batch in self.get_epoch():
                yield batch

    def get_samples(self, n_samples: int):
        """
        Sample for the data, be default we sample batches but subclasses can
        override this method to provide other kinds of sampling (like sampling individual elements).
        Must return both an iterator and the exact size of the iterator.
        """
        return self.get_batches(n_samples), n_samples

    def percent_filtered(self):
        # TODO nicer to just have "unfiltered_size"?
        """ If any filtering was done, the percent of examples that were filtered. Exposed so evaluators
         can compute percentages fairly even if some examples were removed during pre-processing """
        return None

    def __len__(self):
        """ Number of batches per an epoch """
        raise NotImplementedError(self.__class__)


class TrainingData(Configurable):

    def get_train(self) -> Dataset:
        raise NotImplementedError()

    def get_eval(self) -> Dict[str, Dataset]:
        raise NotImplementedError()

    def get_train_corpus(self) -> object:
        """
        Return an object derived from the training data that will be passed to the model's initialization phase,
        what exactly is returned can be arbitrary, but will need to be compatible with
        the model's requirements. Example, return word counts to be used to decide what word vecs to train.
        """
        raise NotImplementedError()

    def get_resource_loader(self) -> ResourceLoader:
        return ResourceLoader()


def shuffle_list_buckets(data, key, rng):
    start = 0
    end = 0
    while start < len(data):
        while end < len(data) and key(data[start]) == key(data[end]):
            end += 1
        rng.shuffle(data[start:end])
        start = end
    return data


class ListBatcher(Configurable):
    def get_epoch(self, data: List):
        raise NotImplementedError()

    def get_fixed_batch_size(self):
        """ Return the batch size if it is constant, else None """
        raise NotImplementedError()

    def get_max_batch_size(self):
        """ Return upper bound on the batch size """
        raise NotImplementedError()

    def epoch_size(self, n_elements):
        raise NotImplementedError()


class FixedOrderBatcher(ListBatcher):
    def __init__(self, batch_size: int, truncate_batches=False):
        self.batch_size = batch_size
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        n_batches = len(data) // self.batch_size
        for i in range(n_batches):
            yield data[i*self.batch_size:(i + 1)*self.batch_size]
        if self.truncate_batches and (len(data) % self.batch_size) > 0:
            yield data[self.batch_size * (len(data) // self.batch_size):]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ShuffledBatcher(ListBatcher):
    def __init__(self,
                 batch_size: int,
                 truncate_batches=False):
        self.batch_size = batch_size
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        data = list(data)
        np.random.shuffle(data)
        n_batches = len(data) // self.batch_size
        for i in range(n_batches):
            yield data[i*self.batch_size:(i + 1)*self.batch_size]
        if self.truncate_batches and (len(data) % self.batch_size) > 0:
            yield data[self.batch_size * (len(data) // self.batch_size):]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ClusteredBatcher(ListBatcher):
    def __init__(self,
                 batch_size: int,
                 clustering: Callable,
                 shuffle_buckets=False,
                 truncate_batches=False):
        self.batch_size = batch_size
        self.clustering = clustering
        self.shuffle_buckets = shuffle_buckets
        self.truncate_batches = truncate_batches

    def get_fixed_batch_size(self):
        return None if self.truncate_batches else self.batch_size

    def get_max_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        data = sorted(data, key=self.clustering)
        if self.shuffle_buckets:
            shuffle_list_buckets(data, self.clustering, np.random)
        n_batches = len(data) // self.batch_size
        intervals = [(i*self.batch_size, (i + 1)*self.batch_size) for i in range(0, n_batches)]
        remainder = len(data) % self.batch_size
        if self.truncate_batches and remainder > 0:
            intervals.append((len(data) - remainder, len(data)))
        np.random.shuffle(intervals)
        for i, j in intervals:
            yield data[i:j]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class ListDataset(Dataset):
    """ Dataset with a fixed list of elements """

    def __init__(self, data: List, batching: ListBatcher, unfiltered_len: Optional[int]=None):
        self.data = data
        self.batching = batching
        self.unfiltered_len = unfiltered_len

    def get_samples(self, n_examples) -> Iterator:
        n_batches = n_examples // self.batching.get_max_batch_size()
        return self.get_batches(n_batches), n_batches

    @property
    def batch_size(self):
        return self.batching.get_fixed_batch_size()

    def get_epoch(self):
        return self.batching.get_epoch(self.data)

    def percent_filtered(self):
        if self.unfiltered_len is None:
            return None
        return (self.unfiltered_len - len(self.data)) / self.unfiltered_len

    def get_n_examples(self):
        return len(self.data)

    def __len__(self):
        return self.batching.epoch_size(len(self.data))

