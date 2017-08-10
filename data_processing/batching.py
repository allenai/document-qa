from typing import List, NamedTuple, Callable, Union, Optional, Iterable

import numpy as np
from numpy.random import RandomState


""" 
Utility method for batching in-memory data 
"""


Batch = NamedTuple("Batch", [("batch_num", int), ("epoch_num", int), ("data", object)])


def get_samples(data: Union[List, Callable], n_epochs=0,
                n_batches=0, rng: Optional[RandomState]=None):
    if rng is None:
        rng = np.random

    if isinstance(data, List):
        data = list(data)  # don't modify the input list
        gen = lambda: data
    else:
        gen = data

    epoch_num = -1
    while n_epochs > 0 or n_batches > 0:
        epoch_num += 1
        epoch = gen()
        rng.shuffle(epoch)
        cutoff = len(epoch)

        if n_epochs > 0:
            n_epochs -= 1
        elif n_batches >= len(epoch):
            n_batches -= len(epoch)
        else:
            cutoff = n_batches
            n_batches = 0

        for i, element in enumerate(data[:cutoff]):
            yield Batch(i, epoch_num, element)


def get_sequential_batches(data: Union[Iterable, Callable], batch_size: int,
                n_epochs=0, n_batches=0):
    if n_epochs == 0 and n_batches == 0:
        raise ValueError()
    if n_epochs < 0 or n_batches < 0 or batch_size <= 0:
        raise ValueError()

    if isinstance(data, Iterable):
        gen = lambda: data
    else:
        gen = data

    epoch_num = 0
    while n_epochs > 0 or n_batches > 0:
        batch_number = 0
        if n_epochs > 0:
            n_epochs -= 1
            for element in gen():
                yield Batch(epoch_num, batch_number, element)
                batch_number += 1
        else:
            for element in gen():
                yield Batch(epoch_num, batch_number, element)
                batch_number += 1
                n_batches -= 1
                if n_batches == 0:
                    return



def get_batches(data: Union[List, Callable], batch_size: int,
                n_epochs=0, n_batches=0,
                rng: Optional[RandomState]=None, allow_truncate=True,
                shuffle=True):
    if n_epochs == 0 and n_batches == 0:
        raise ValueError()
    if n_epochs < 0 or n_batches < 0 or batch_size <= 0:
        raise ValueError()

    if rng is None and shuffle:
        rng = np.random

    if isinstance(data, List):
        static_data = list(data)
        gen = lambda: static_data
    else:
        gen = data

    epoch_num = 0
    while n_epochs > 0 or n_batches > 0:
        epoch = gen()
        if shuffle:
            rng.shuffle(epoch)
        epoch_batches = len(epoch) // batch_size
        remainder = allow_truncate and (len(epoch) % batch_size > 0)

        if n_epochs > 0:
            n_epochs -= 1
        else:
            if epoch_batches+remainder <= n_batches:
                # Fall back to yielding a complete epoch
                n_batches -= epoch_batches + remainder
            else:
                # Yield a partial epoch, do not truncate
                for batch_ix in range(n_batches):
                    yield Batch(batch_ix, epoch_num, epoch[batch_ix * batch_size:(batch_ix + 1) * batch_size])
                return

        # Yield an entire epoch
        for batch_ix in range(epoch_batches):
            yield Batch(batch_ix, epoch_num, epoch[batch_ix*batch_size:(batch_ix+1)*batch_size])
        if remainder:
            yield Batch(n_batches, epoch_num, epoch[epoch_batches*batch_size:])
        epoch_num += 1


def get_batches_ordered(data: Union[List, Callable], batch_size,
                        n_epochs=0, n_batches=0,
                        rng: Optional[RandomState]=None, allow_truncate=True):
    if n_epochs == 0 and n_batches == 0:
        raise ValueError()
    if n_epochs < 0 or n_batches < 0 or batch_size <= 0:
        raise ValueError()

    if rng is None:
        rng = np.random

    if isinstance(data, List):
        static_data = list(data)
        gen = lambda: static_data
    else:
        gen = data

    epoch_num = 0
    while n_epochs > 0 or n_batches > 0:
        epoch = gen()
        epoch_batches = len(epoch) // batch_size

        remainder = allow_truncate and (len(epoch) % batch_size > 0)
        intervals = [(i*batch_size, (i+1)*batch_size) for i in range(epoch_batches)]
        if remainder:
            # Add a partial batch to cover the entire epoch
            # TODO better to randomly decrease some of the existing batches to keep the sizes similar
            intervals.append((epoch_batches*batch_size, len(epoch)))
        else:
            # randomly select elements to get truncated
            delete = rng.choice(len(epoch), len(epoch) % batch_size, replace=False)
            data = np.delete(data, delete)

        rng.shuffle(intervals)

        if n_epochs > 0:
            n_epochs -= 1
        else:
            if epoch_batches + remainder <= n_batches:
                # Fall back to yielding a complete epoch
                n_batches -= epoch_batches + remainder
            else:
                # Yield a partial epoch
                for batch_ix, (start,end) in enumerate(intervals[:n_batches]):
                    yield Batch(batch_ix, epoch_num, epoch[start:end])
                return

        # Yield an entire epoch
        for batch_ix, (start, end) in enumerate(intervals):
            yield Batch(batch_ix, epoch_num, epoch[start:end])
        epoch_num += 1


def shuffle_list_buckets(data, key, rng):
    start = 0
    end = 0
    while start < len(data):
        while end < len(data) and key(data[start]) == key(data[end]):
            end += 1
        rng.shuffle(data[start:end])
        start = end
    return data


def get_clustered_batches(data: Union[List, Callable], batch_size,
                          key,  n_epochs=0, n_batches=0,
                          shuffle_buckets=True,
                          rng: Optional[RandomState]=None, allow_truncate=True):
    if rng is None:
        rng = np.random

    # Use sorted data
    if isinstance(data, List):
        sorted_gen = sorted(data, key=key)
    else:
        sorted_gen = lambda: sorted(data(), key=key)

    # and optionally shuffle the same-valued elements each epoch
    if shuffle_buckets:
        if isinstance(data, List):
            shuf = lambda: shuffle_list_buckets(sorted_gen, key, rng)
        else:
            shuf = lambda: shuffle_list_buckets(sorted_gen(), key, rng)
    else:
        shuf = sorted_gen

    return get_batches_ordered(shuf, batch_size, n_epochs, n_batches, rng=rng, allow_truncate=allow_truncate)



