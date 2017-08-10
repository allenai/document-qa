import gzip
from os.path import join, exists
from typing import Union, Iterable, Optional

import numpy as np

from config import VEC_DIR


def _vec_path(vec_name):
    return join(VEC_DIR, vec_name + ".txt")


def load_word_vectors(vec_name: str, vocab: Optional[Iterable[str]]=None):
    if vocab is not None:
        vocab = set(x.lower() for x in vocab)
    pruned_dict = {}
    vec_path = _vec_path(vec_name)
    if not exists(vec_path):
        vec_path += "gz"
        if not exists(vec_path):
            raise ValueError("Could not find word vectors: %s at %s" % (vec_name, vec_path))
        handle = lambda x: gzip.open(x, 'r', encoding='utf-8', errors='ignore')
    else:
        handle = lambda x: open(x, 'r', encoding='utf-8', errors='ignore')

    # some of the large vec files produce utf-8 errors for some words, just skip them
    with handle(vec_path) as fh:
        for line in fh:
            word_ix = line.find(" ")
            word = line[:word_ix]
            if (vocab is None) or (word.lower() in vocab):
                pruned_dict[word] = np.array([float(x) for x in line[word_ix + 1:-1].split(" ")], dtype=np.float32)
    return pruned_dict
