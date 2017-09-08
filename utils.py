import argparse
from datetime import datetime
from json import JSONEncoder

from os.path import join
from typing import List, TypeVar, Iterable

from tqdm import tqdm

from data_processing.word_vectors import load_word_vectors
import numpy as np


class ResourceLoader(object):
    """
    Abstraction for models the need access to external resources to setup, currently just
    for word-vectors.
    """

    def __init__(self, load_vec_fn=load_word_vectors):
        self.load_vec_fn = load_vec_fn

    def load_word_vec(self, vec_name, voc=None):
        return self.load_vec_fn(vec_name, voc)


class DummyResourceLoader(ResourceLoader):
    def load_word_vec(self, vec_name):
        return {"the": np.zeros(100, dtype=np.float32)}


class CachingResourceLoader(ResourceLoader):

    def __init__(self, load_vec_fn=load_word_vectors):
        super().__init__(load_vec_fn)
        self.word_vec = {}

    def load_word_vec(self, vec_name):
        if vec_name not in self.word_vec:
            self.word_vec[vec_name] = super().load_word_vec(vec_name)
        return self.word_vec[vec_name]


def print_table(table: List[List[str]]):
    col_lens = [0] * len(table[0])
    for row in table:
        for i,cell in enumerate(row):
            col_lens[i] = max(len(cell), col_lens[i])

    # pads string to the requested length
    formats = ["{0:<%d}" % x for x in col_lens]
    for row in table:
        print(" ".join(formats[i].format(row[i]) for i in range(len(row))))


def transpose_lists(lsts):
    return [list(i) for i in zip(*lsts)]


def get_output_name_from_cli():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', '-n', nargs=1, help='name of output to exmaine')

    args = parser.parse_args()
    if args.name:
        out = join(args.name[0] + "-" + datetime.now().strftime("%m%d-%H%M%S"))
        print("Starting run on: " + out)
    else:
        out = "out/run-" + datetime.now().strftime("%m%d-%H%M%S")
        print("Starting run on: " + out)
    return out


def max_or_none(a, b):
    if a is None or b is None:
        return None
    return max(a, b)


T = TypeVar('T')


def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]


def split(lst, n_groups):
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst: List, max_group_size) -> List[List]:
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst)+max_group_size-1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups
