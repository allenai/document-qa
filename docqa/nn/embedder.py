from collections import Counter
from typing import List, Iterable

import numpy as np
import tensorflow as tf

from docqa.configurable import Configurable
from docqa.nn.layers import Encoder
from docqa.utils import ResourceLoader

"""
Classes for embedding words/chars
"""


class WordEmbedder(Configurable):
    """
    Responsible for mapping words -> ids, ids -> words, and ids -> embeddings
    matrices. Needs to be initialized from a corpus by `set_vocab` after construction
    """

    def set_vocab(self, corpus, word_vec_loader: ResourceLoader, special_tokens: List[str]):
        raise NotImplementedError()

    def is_vocab_set(self):
        raise NotImplementedError()

    def question_word_to_ix(self, word, is_train) -> int:
        raise NotImplementedError()

    def context_word_to_ix(self, word, is_train) -> int:
        raise NotImplementedError()

    def query_once(self) -> bool:
        """
        Should the embedder be queried once for each unique word in the input, or once for each word.
        Intended to support placeholders, although I ended up not experimenting much w/that route
        """
        return False

    def init(self, word_vec_loader, voc: Iterable[str]):
        raise NotImplementedError()

    def embed(self, is_train, *word_ix_and_mask):
        """ [(word_ix, mask)...] -> [word_embed, ...]"""
        raise NotImplemented()


class CharEmbedder(Configurable):
    """
    Responsible for mapping char -> ids, ids -> char, and ids -> embeddings
    Needs to be initialized from a corpus by `set_vocab` after construction
    """

    def set_vocab(self, corpus):
        raise NotImplementedError()

    def get_word_size_th(self):
        raise ValueError()

    def char_to_ix(self, char):
        raise NotImplementedError()

    def init(self, word_vec_loader, voc: Iterable[str]):
        raise NotImplementedError()

    def embed(self, is_train, *char_ix_and_mask):
        """ [(char_ix, mask)...] -> [word_embed, ...]"""
        raise NotImplemented()


class LearnedCharEmbedder(CharEmbedder):

    def __init__(self, word_size_th: int, char_th: int, char_dim: int,
                 init_scale: float=0.1, force_cpu: bool=False):
        self.force_cpu = force_cpu
        self.word_size_th = word_size_th
        self.char_th = char_th
        self.char_dim = char_dim
        self.init_scale = init_scale

        self._char_to_ix = None

    def get_word_size_th(self):
        return self.word_size_th

    def set_vocab(self, corpus):
        w_counts = corpus.get_word_counts()
        counts = Counter()
        for w,count in w_counts.items():
            for c in w:
                counts[c] += count

        self._char_to_ix = {c:i+2 for i,c in enumerate(c for c,count in counts.items() if count >= self.char_th)}
        print("Learning an embedding for %d characters" % (len(self._char_to_ix)-1))

    def init(self, word_vec_loader, voc: Iterable[str]):
        pass

    def char_to_ix(self, char):
        return self._char_to_ix.get(char, 1)

    def embed(self, is_train, *char_ix):
        if self.force_cpu:
            with tf.device('/cpu:0'):
                return self._embed(*char_ix)
        else:
            return self._embed(*char_ix)

    def _embed(self, *char_ix):
        zero = tf.zeros((1, self.char_dim), dtype=np.float32)
        mat = tf.get_variable("char_emb_mat", (len(self._char_to_ix)+1, self.char_dim),
                              tf.float32, initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
        emb_mat = tf.concat([zero, mat], axis=0)

        x = char_ix[0]
        tf.nn.embedding_lookup(emb_mat, x[0])

        return [tf.nn.embedding_lookup(emb_mat, x[0]) for x in char_ix]

    def __setstate__(self, state):
        if "state" in state:
            if "force_cpu" not in state["state"]:
                state["state"]["force_cpu"] = False
        super().__setstate__(state)


class CharWordEmbedder(Configurable):
    """
    Derives word embeddings from character embeddings by combining a character embedder and a reduce layer
    """
    def __init__(self, embedder: CharEmbedder, layer: Encoder,
                 shared_parameters: bool):
        self.embeder = embedder
        self.layer = layer
        self.shared_parameters = shared_parameters

    def embed(self, is_train, *char_ix):
        embeds = self.embeder.embed(is_train, *char_ix)
        if self.shared_parameters:
            with tf.variable_scope("embedding"):
                output = [self.layer.apply(is_train, embeds[0], char_ix[0][1])]
            with tf.variable_scope("embedding", reuse=True):
                for i in range(1, len(embeds)):
                    output.append(self.layer.apply(is_train, embeds[i], char_ix[i][1]))
        else:
            output = []
            for i, emb in enumerate(embeds):
                with tf.variable_scope("embedding%d_%s" % (i, emb.name)):
                    output.append(self.layer.apply(is_train, emb, char_ix[i][1]))
        return output

    def __setstate__(self, state):
        if "state" in state:
            state["state"]["version"] = state["version"]
            state = state["state"]
        if "share" in state:
            state["shared_parameters"] = state["share"]
            del state["share"]
        super().__setstate__(state)


def shrink_embed(mat, word_ixs: List):
    """
    Build an embedding matrix that contains only the elements in `word_ixs`,
    and map `word_ixs` to tensors the index into they new embedding matrix.
    Useful if you want to dropout the embeddings w/o dropping out the entire matrix
    """
    all_words, out_id = tf.unique(tf.concat([tf.reshape(x, (-1,)) for x in word_ixs], axis=0))
    mat = tf.gather(mat, all_words)
    partitions = tf.split(out_id, [tf.reduce_prod(tf.shape(x)) for x in word_ixs])
    partitions = [tf.reshape(x, tf.shape(o)) for x,o in zip(partitions, word_ixs)]
    return mat, partitions


class FixedWordEmbedder(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0.05,
                 learn_unk: bool = True,
                 keep_probs: float = 1,
                 keep_word: float= 1,
                 shrink_embed: bool=False,
                 cpu=False):
        self.keep_word = keep_word
        self.keep_probs = keep_probs
        self.word_vec_init_scale = word_vec_init_scale
        self.learn_unk = learn_unk
        self.vec_name = vec_name
        self.cpu = cpu
        self.shrink_embed = shrink_embed

        # Built in `init`
        self._word_to_ix = None
        self._word_emb_mat = None
        self._special_tokens = None

    def set_vocab(self, _, loader: ResourceLoader, special_tokens: List[str]):
        if special_tokens is not None:
            self._special_tokens = sorted(special_tokens)

    def is_vocab_set(self):
        return True

    def question_word_to_ix(self, word, is_train):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def context_word_to_ix(self, word, is_train):
        # print(word)
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    @property
    def version(self):
        # added `cpu`
        return 1

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        if self.cpu:
            with tf.device("/cpu:0"):
                self._init(loader, voc)
        else:
            self._init(loader, voc)

    def _init(self, loader: ResourceLoader, voc: Iterable[str]):
        # TODO we should not be building variables here
        if voc is not None:
            word_to_vec = loader.load_word_vec(self.vec_name, voc)
        else:
            word_to_vec = loader.load_word_vec(self.vec_name)
            voc = set(word_to_vec.keys())

        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]

        null_embed = tf.zeros((1, dim), dtype=tf.float32)
        unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
                                    dtype=np.float32, trainable=self.learn_unk,
                                    initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                              self.word_vec_init_scale))
        ix = 2
        matrix_list = [null_embed, unk_embed]

        if self._special_tokens is not None and len(self._special_tokens) > 0:
            print("Building embeddings for %d special_tokens" % (len(self._special_tokens)))
            tok_embed = tf.get_variable(shape=(len(self._special_tokens), dim), name="token_embed",
                                        dtype=np.float32, trainable=True,
                                        initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                                  self.word_vec_init_scale))
            matrix_list.append(tok_embed)
            for token in self._special_tokens:
                self._word_to_ix[token] = ix
                ix += 1

        mat = []
        for word in voc:
            if word in self._word_to_ix:
                continue  # in case we already added due after seeing a capitalized version of `word`
            if word in word_to_vec:
                mat.append(word_to_vec[word])
                self._word_to_ix[word] = ix
                ix += 1
            else:
                lower = word.lower()  # Full back to the lower-case version
                if lower in word_to_vec and lower not in self._word_to_ix:
                    mat.append(word_to_vec[lower])
                    self._word_to_ix[lower] = ix
                    ix += 1

        print("Had pre-trained word embeddings for %d of %d words" % (len(mat), len(voc)))

        matrix_list.append(tf.constant(value=np.vstack(mat)))

        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def embed(self, is_train, *word_ix):
        if any(len(x) != 2 for x in word_ix):
            raise ValueError()
        mat = self._word_emb_mat

        if self.keep_probs < 1:
            mat = tf.cond(is_train,
                          lambda: tf.nn.dropout(mat, self.keep_probs),
                          lambda: mat)
        if self.keep_word < 1:
            mat = tf.cond(is_train,
                          lambda: tf.nn.dropout(mat, self.keep_word, (mat.shape.as_list()[0], 1)),
                          lambda: mat)
        if self.cpu:
            with tf.device("/cpu:0"):
                return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
        else:
            return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)

    def __setstate__(self, state):
        if "state" in state:
            if "cpu" not in state["state"]:
                state["state"]["cpu"] = False
            if "keep_probs" not in state["state"]:
                state["state"]["keep_probs"] = 1.0
            if "keep_word" not in state["state"]:
                state["state"]["keep_word"] = 1.0
            if "_special_tokens" not in state["state"]:
                state["state"]["_special_tokens"] = []
        super().__setstate__(state)


class FixedWordEmbedderPlaceholders(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0.05,
                 keep_probs: float = 1,
                 keep_word: float= 1,
                 n_placeholders: int=1000,
                 placeholder_stddev: float=0.5,
                 placeholder_flag: bool=False,
                 cpu=False):
        self.placeholder_stddev = placeholder_stddev
        self.placeholder_flag = placeholder_flag
        self.keep_word = keep_word
        self.keep_probs = keep_probs
        self.word_vec_init_scale = word_vec_init_scale
        self.vec_name = vec_name
        self.cpu = cpu
        self.n_placeholders = n_placeholders
        self._on_placeholder = 0
        self._placeholder_start = None

        # Built in `init`
        self._word_to_ix = None
        self._word_emb_mat = None
        self._special_tokens = None

    def set_vocab(self, _, loader: ResourceLoader, special_tokens: List[str]):
        if special_tokens is not None:
            self._special_tokens = sorted(special_tokens)

    def is_vocab_set(self):
        return True

    def query_once(self) -> bool:
        return True

    def question_word_to_ix(self, word, is_train):
        ix = self._word_to_ix.get(word)
        if ix is None:
            ix = self._word_to_ix.get(word.lower())
            if ix is None:
                self._on_placeholder = (self._on_placeholder + 1) % self.n_placeholders
                ix = self._placeholder_start + self._on_placeholder
        return ix

    def context_word_to_ix(self, word, is_train):
        ix = self._word_to_ix.get(word)
        if ix is None:
            ix = self._word_to_ix.get(word.lower())
            if ix is None:
                self._on_placeholder = (self._on_placeholder + 1) % self.n_placeholders
                ix = self._placeholder_start + self._on_placeholder
        return ix

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        if self.cpu:
            with tf.device("/cpu:0"):
                self._init(loader, voc)
        else:
            self._init(loader, voc)

    def _init(self, loader: ResourceLoader, voc: Iterable[str]):
        # TODO we should not be building variables here
        if voc is not None:
            word_to_vec = loader.load_word_vec(self.vec_name, voc)
        else:
            word_to_vec = loader.load_word_vec(self.vec_name)
            voc = set(word_to_vec.keys())

        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]
        if self.placeholder_flag:
            dim += 1

        null_embed = tf.zeros((1, dim), dtype=tf.float32)
        ix = 1
        matrix_list = [null_embed]

        if self._special_tokens is not None and len(self._special_tokens) > 0:
            print("Building embeddings for %d special_tokens" % (len(self._special_tokens)))
            tok_embed = tf.get_variable(shape=(len(self._special_tokens), dim), name="token_embed",
                                        dtype=np.float32, trainable=True,
                                        initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                                  self.word_vec_init_scale))
            matrix_list.append(tok_embed)
            for token in self._special_tokens:
                self._word_to_ix[token] = ix
                ix += 1

        mat = []
        for word in voc:
            if word in self._word_to_ix:
                continue  # in case we already added due after seeing a capitalized version of `word`
            if word in word_to_vec:
                mat.append(word_to_vec[word])
                self._word_to_ix[word] = ix
                ix += 1
            else:
                lower = word.lower()  # Full back to the lower-case version
                if lower in word_to_vec and lower not in self._word_to_ix:
                    mat.append(word_to_vec[lower])
                    self._word_to_ix[lower] = ix
                    ix += 1

        print("Had pre-trained word embeddings for %d of %d words" % (len(mat), len(voc)))

        mat = np.vstack(mat)
        if self.placeholder_flag:
            mat = np.concatenate([mat, np.zeros((len(mat), 1), dtype=np.float32)], axis=1)
        matrix_list.append(tf.constant(value=mat))

        self._placeholder_start = ix

        if self.placeholder_flag:
            def init(shape, dtype=None, partition_info=None):
                out = tf.random_normal((self.n_placeholders, dim - 1), stddev=self.placeholder_stddev)
                return tf.concat([out, tf.ones((self.n_placeholders, 1))], axis=1)
            init_fn = init
        else:
            init_fn = tf.random_normal_initializer(stddev=self.placeholder_stddev)

        matrix_list.append(tf.get_variable("placeholders", (self.n_placeholders, mat.shape[1]),
                                           tf.float32, trainable=False,
                                           initializer=init_fn))

        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def embed(self, is_train, *word_ix):
        if any(len(x) != 2 for x in word_ix):
            raise ValueError()
        if self.cpu:
            with tf.device("/cpu:0"):
                return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
        else:
            return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
