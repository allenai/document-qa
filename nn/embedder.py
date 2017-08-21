from collections import Counter
from typing import List, Iterable, Optional

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

from configurable import Configurable
from data_processing.paragraph_qa import ParagraphAndQuestion
from nn.layers import SqueezeLayer, Updater, Encoder
from utils import ResourceLoader

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

    def question_word_to_ix(self, word, is_train):
        raise NotImplementedError()

    def context_word_to_ix(self, word, is_train):
        raise NotImplementedError()

    def init(self, word_vec_loader, voc: Iterable[str]):
        raise NotImplementedError()

    def embed(self, is_train, *word_ix):
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

    def embed(self, is_train, *char_ix):
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
        self.share = shared_parameters

    def embed(self, is_train, *char_ix):
        embeds = self.embeder.embed(is_train, *char_ix)
        if self.share:
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

    def set_vocab(self, data: List[ParagraphAndQuestion], loader: ResourceLoader, special_tokens: List[str]):
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
        word_to_vec = loader.load_word_vec(self.vec_name, voc)
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
            print("Building embedings for %d special_tokens" % (len(self._special_tokens)))
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

        print("Had pre-trained word embeddings for %d of %d examples" % (len(mat), len(voc)))

        matrix_list.append(tf.constant(value=np.vstack(mat)))

        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def embed(self, is_train, *word_ix):
        if any(len(x) != 2 for x in word_ix):
            raise ValueError()
        mat = self._word_emb_mat

        # if self.keep_word < 1 or self.keep_probs < 1 and self.shrink_embed:
        #     words, masks = zip(*word_ix)
        #     mat, words = shrink_embed(mat, words)
        #     word_ix = zip(words, masks)

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
        if "cpu" not in state["state"]:
            state["state"]["cpu"] = False
        if "keep_probs" not in state["state"]:
            state["state"]["keep_probs"] = 1.0
        if "keep_word" not in state["state"]:
            state["state"]["keep_word"] = 1.0
        if "_special_tokens" not in state["state"]:
            state["state"]["_special_tokens"] = []
        super().__setstate__(state)


class PartialTrainEmbedder(WordEmbedder, Configurable):

    def __init__(self,
                 vec_name: str,
                 n_words: int = 1000,
                 word_vec_init_scale: float = 0.05,
                 unk_init_scale: float = 0.0,
                 learn_unk: bool = True,
                 train_unknown: bool = True):
        self.word_vec_init_scale = word_vec_init_scale
        self.unk_init_scale = unk_init_scale
        self.learn_unk = learn_unk
        self.train_unknown = train_unknown
        self.vec_name = vec_name
        self.n_words = n_words

        # Words/Chars we will learn embeddings for
        self._train_words = None

        # Built in `init`
        self._word_to_ix = None
        self._word_emb_mat = None

    def set_vocab(self, corpus, word_vec_loader: ResourceLoader, special_tokens: List[str]):
        quesiton_counts = corpus.get_question_counts()
        lower_counts = Counter()
        for word, c, in quesiton_counts:
            lower_counts[word.lower()] += c
        self._train_words = lower_counts.most_common(self.n_words)

    def question_word_to_ix(self, word):
        return self._word_to_ix.get(word.lower(), 1)

    def context_word_to_ix(self, word):
        return self._word_to_ix.get(word.lower(), 1)

    def is_vocab_set(self):
        raise NotImplementedError()

    def init(self, word_vec_loader, voc: Iterable[str]):
        word_to_vec = word_vec_loader.get_word_vecs(self.vec_name)
        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]

        null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
        unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
                                    dtype=np.float32, trainable=self.learn_unk,
                                    initializer=tf.random_uniform_initializer(-self.unk_init_scale,
                                                                              self.unk_init_scale))
        train_embed = tf.get_variable(shape=(len(self._train_words), dim), name="train_words", dtype=np.float32,
                                      initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                                self.word_vec_init_scale))
        matrix_list = [null_embed, unk_embed, train_embed]
        ix = 2
        for word in self._train_words:
            self._word_to_ix[word] = ix
            ix += 1

        train_word_set = set(self._train_words)

        fixed_mat = []
        for word in voc:
            if word in self._word_to_ix:
                continue
            wl = word.lower()
            if wl in train_word_set:
                continue
            if word in word_to_vec:
                fixed_mat.append(word_to_vec[word])
                self._word_to_ix[word] = ix
                ix += 1
            else:
                if wl in word_to_vec and wl not in self._word_to_ix:
                    fixed_mat.append(word_to_vec[wl])
                    self._word_to_ix[wl] = ix
                    ix += 1

        matrix_list.append(fixed_mat)
        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)


class DropNames(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 selector,
                 word_vec_init_scale: float = 0,
                 keep_probs: float=0.5,
                 batch: bool=True,
                 swapped_flag: bool = False,
                 kind: str="shuffle",
                 learn_unk: bool = False):
        self.swapped_flag = swapped_flag
        self.kind = kind
        self.batch = batch
        self.keep_probs = keep_probs
        self.learn_unk = learn_unk
        self.word_vec_init_scale = word_vec_init_scale
        self.vec_name = vec_name
        self.selector = selector

        # corpus stats we need to keep around
        self._name_vecs_start = 0
        self._name_vecs_end = 0
        self._swap_start = None
        self._swap_end = None

        self._word_to_ix = None
        self._word_emb_mat = None

    def set_vocab(self, corpus, loader: ResourceLoader, special_tokens):
        if special_tokens is not None and len(special_tokens) > 0:
            raise NotImplementedError()
        self.selector.init(corpus.get_word_counts())

    def is_vocab_set(self):
        return True

    def question_word_to_ix(self, word, is_train):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def context_word_to_ix(self, word, is_train):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        with tf.device("/cpu:0"):
            word_to_vec = loader.load_word_vec(self.vec_name)
            self._word_to_ix = {}

            dim = next(iter(word_to_vec.values())).shape[0]

            null_embed = tf.zeros((1, dim), dtype=tf.float32)
            unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
                                        dtype=tf.float32, trainable=self.learn_unk,
                                        initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                                  self.word_vec_init_scale))
            matrix_list = [null_embed, unk_embed]
            ix = len(matrix_list)

            mat = []
            names = []
            for word in voc:
                if word in self._word_to_ix:
                    continue  # in case we already added due after seeing a capitalized version of `word`
                if self.selector.select(word):
                    names.append(word)
                    continue
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
            matrix_list.append(tf.constant(np.array(mat)))

            self._name_vecs_start = ix
            name_unk = tf.get_variable(shape=(1, dim), name="name_unk",
                                       dtype=tf.float32,  initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                                  self.word_vec_init_scale))
            matrix_list.append(name_unk)
            ix += 1
            name_mat = []
            for name in names:
                vec = word_to_vec.get(name)
                if vec is not None:
                    name_mat.append(vec)
                    self._word_to_ix[name] = ix
                    ix += 1
                else:
                    word_to_vec[name] = self._name_vecs_start  # unk name
            matrix_list.append(tf.constant(np.array(name_mat)))
            self._name_vecs_end = ix

            print("Have %d named (and %d name vecs), %d other" % (
                len(names), len(name_mat), len(mat)))

            if self.swapped_flag:
                szs = [x.shape.as_list()[0] for x in matrix_list]
                self._swap_start = sum(szs)
                self._swap_end = self._swap_start + szs[-1]
                matrix_list.append(matrix_list[-1])
                flags = tf.concat([tf.zeros(self._swap_start),
                                   tf.ones(szs[-1])], axis=0)
                flags = tf.expand_dims(flags, 1)
                self._word_emb_mat = tf.concat(matrix_list, axis=0)
                self._word_emb_mat = tf.concat([self._word_emb_mat, flags], axis=1)
            else:
                self._swap_start = self._name_vecs_start + 1
                self._swap_end = self._name_vecs_end
                self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def _drop_names(self, ix):
        unique_words, unique_idx = tf.unique(ix)
        u_words = tf.shape(unique_words)[0]
        to_drop = tf.logical_and(unique_words >= self._name_vecs_start,
                                 tf.random_uniform((u_words,)) > self.keep_probs)
        if self.kind == "name_unk":
            replace = tf.fill((u_words,), self._name_vecs_start)
        elif self.kind == "unk":
            replace = tf.ones((u_words,), dtype=tf.int32)
        elif self.kind == "shuffle":
            replace = tf.random_uniform(tf.shape(unique_words), self._swap_start,
                                        self._swap_end, dtype=tf.int32)
        else:
            raise NotImplementedError()
        unique_words = tf.where(to_drop, replace, unique_words)
        return tf.gather(unique_words, unique_idx)

    def drop_names_batch(self, words):
        all_words = tf.concat(words, axis=1)
        dropped = self._drop_names(tf.reshape(all_words, (-1, )))
        dropped = tf.reshape(dropped, tf.shape(all_words))
        out = tf.split(dropped, [tf.shape(x)[1] for x in words], axis=1, num=len(words))
        return out

    def drop_names(self, words):
        all_words = tf.concat(words, axis=1)
        dropped_words = tf.map_fn(self._drop_names, all_words)
        dropped_words = tf.split(dropped_words, [tf.shape(x)[1] for x in words], axis=1, num=len(words))
        return dropped_words

    def embed(self, is_train, *word_ix):
        with tf.device("/cpu:0"):
            words, _ = zip(*word_ix)
            words = tf.cond(is_train,
                            lambda: self.drop_names_batch(words) if self.batch else self.drop_names(words),
                            lambda: words)
            return [tf.nn.embedding_lookup(self._word_emb_mat, x) for x in words]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)

    def __setstate__(self, state):
        super().__setstate__(state)



# class DropShuffleNames(WordEmbedder):
#
#     def __init__(self,
#                  vec_name: str,
#                  word_vec_init_scale: float = 0,
#                  name_keep_probs: float=0.5,
#                  learn_unk: bool = False,
#                  named_thresh=0.9):
#         self.name_keep_probs = name_keep_probs
#         self.learn_unk = learn_unk
#         self.word_vec_init_scale = word_vec_init_scale
#         self.vec_name = vec_name
#         self.named_thresh = named_thresh
#
#         # corpus stats we need to keep around
#         self._word_counts = None
#         self._word_counts_lower = None
#         self._stop = None
#         self._name_vecs_start = 0
#         self._name_vecs_end = 0
#
#         self._word_to_ix = None
#         self._word_emb_mat = None
#
#     def set_vocab(self, corpus, loader: ResourceLoader, special_tokens):
#         if special_tokens is not None and len(special_tokens) > 0:
#             raise ValueError()
#         self._word_counts = corpus.get_context_counts()
#         self._word_counts_lower = Counter()
#         for k,v in self._word_counts.items():
#             self._word_counts_lower[k.lower()] += v
#         self._stop = set(stopwords.words('english'))
#
#     def is_named(self, word):
#         if word[0].isupper() and word[1:].islower():
#             wl = word.lower()
#             if wl not in self._stop:
#                 lc = self._word_counts_lower[wl]
#                 if lc == 0 or (self._word_counts[word] / lc) > self.named_thresh:
#                     return True
#         return False
#
#     def is_vocab_set(self):
#         return True
#
#     def question_word_to_ix(self, word):
#         ix = self._word_to_ix.get(word, 1)
#         if ix == 1:
#             return self._word_to_ix.get(word.lower(), 1)
#         else:
#             return ix
#
#     def context_word_to_ix(self, word):
#         ix = self._word_to_ix.get(word, 1)
#         if ix == 1:
#             return self._word_to_ix.get(word.lower(), 1)
#         else:
#             return ix
#
#     def init(self, loader: ResourceLoader, voc: Iterable[str]):
#         word_to_vec = loader.load_word_vec(self.vec_name)
#         self._word_to_ix = {}
#
#         dim = next(iter(word_to_vec.values())).shape[0]
#
#         null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
#         unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
#                                     dtype=np.float32, trainable=self.learn_unk,
#                                     initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
#                                                                               self.word_vec_init_scale))
#         matrix_list = [null_embed, unk_embed]
#         ix = len(matrix_list)
#
#         mat = []
#         names = []
#         for word in voc:
#             if word in self._word_to_ix:
#                 continue  # in case we already added due after seeing a capitalized version of `word`
#             if self.is_named(word):
#                 names.append(word)
#                 continue
#             if word in word_to_vec:
#                 mat.append(word_to_vec[word])
#                 self._word_to_ix[word] = ix
#                 ix += 1
#             else:
#                 lower = word.lower()  # Full back to the lower-case version
#                 if lower in word_to_vec and lower not in self._word_to_ix:
#                     mat.append(word_to_vec[lower])
#                     self._word_to_ix[lower] = ix
#                     ix += 1
#         matrix_list.append(tf.constant(np.array(mat, dtype=np.float32)))
#
#         self._name_vecs_start = ix
#         name_unk = tf.get_variable(shape=(1, dim), name="name_unk",
#                                    dtype=np.float32,  initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
#                                                                               self.word_vec_init_scale))
#         matrix_list.append(name_unk)
#         ix += 1
#         name_mat = []
#         for name in names:
#             vec = word_to_vec.get(name)
#             if vec is not None:
#                 name_mat.append(vec)
#                 self._word_to_ix[name] = -ix
#                 ix += 1
#             else:
#                 word_to_vec[name] = self._name_vecs_start  # unk for names
#         print(len(names))
#         matrix_list.append(tf.constant(np.array(name_mat, dtype=np.float32)))
#         self._name_vecs_end = ix
#
#         print("Have %d named (and %d name vecs), %d other" % (
#             len(names), len(name_mat), len(mat)))
#
#         self._word_emb_mat = tf.concat(matrix_list, axis=0)
#
#     def get_placeholder(self, ix, is_train):
#         if not is_train or np.random.random() < self.name_keep_probs:
#             return abs(ix)
#         return np.random.randint(self._name_vecs_start, self._name_vecs_end)
#
#     def embed(self, is_train, *word_ix):
#         return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
#
#     def __getstate__(self):
#         state = dict(self.__dict__)
#         state["_word_emb_mat"] = None  # we will rebuild these anyway
#         state["_word_to_ix"] = None
#         return dict(version=self.version, state=state)
#
#     def __setstate__(self, state):
#         super().__setstate__(state)


class SelectivePlaceholder(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0,
                 keep_probs: float = 1,
                 n_placeholder_rot=150,
                 place_holder_scale = 0.5,
                 placeholder_mapper: Optional[Updater]=None,
                 placeholder_dist=None,
                 named_thresh=0.9):
        self.keep_probs = keep_probs
        self.n_placeholder_rot = n_placeholder_rot
        self.word_vec_init_scale = word_vec_init_scale
        self.placeholder_mapper = placeholder_mapper
        self.vec_name = vec_name
        self.place_holder_scale = place_holder_scale
        self.named_thresh = named_thresh
        self.placeholder_dist = placeholder_dist

        # corpus stats we need to keep around
        self._word_counts = None
        self._word_counts_lower = None
        self._stop = None

        # Built in `init`
        self._unk_ix = 0
        self._num_ix = 0
        self._name_ix = 0

        self._word_to_ix = None
        self._word_emb_mat = None
        self._n_vecs = None

    def set_vocab(self, corpus, loader: ResourceLoader, special_tokens):
        if special_tokens is not None and len(special_tokens) > 0:
            raise ValueError()
        self._word_counts = corpus.get_context_counts()
        self._word_counts_lower = Counter()
        for k,v in self._word_counts.items():
            self._word_counts_lower[k.lower()] += v
        self._stop = set(stopwords.words('english'))

    def is_named(self, word):
        if word[0].isupper() and word[1:].islower():
            wl = word.lower()
            if wl not in self._stop:
                lc = self._word_counts_lower[wl]
                if lc == 0 or (self._word_counts[word] / lc) > self.named_thresh:
                    return True
        return False

    def is_vocab_set(self):
        return True

    def question_word_to_ix(self, word):
        ix = self._word_to_ix.get(word, -3)
        if ix == -3:
            return self._word_to_ix.get(word.lower(), -3)
        else:
            return ix

    def context_word_to_ix(self, word):
        ix = self._word_to_ix.get(word, -3)
        if ix == -3:
            return self._word_to_ix.get(word.lower(), -3)
        else:
            return ix

    @property
    def word_embed_mat(self):
        return self._word_emb_mat

    def get_placeholder(self, ix, is_train):
        if ix == -1:
            if self._num_ix+1 >= self.n_placeholder_rot:
                self._num_ix = 0
            else:
                self._num_ix += 1
            return self._n_vecs + self._num_ix
        elif ix == -2:
            if self._name_ix+1 >= self.n_placeholder_rot:
                self._name_ix = 0
            else:
                self._name_ix += 1
            return self._n_vecs + self.n_placeholder_rot + self._name_ix
        elif ix == -3:
            if self._unk_ix+1 >= self.n_placeholder_rot:
                self._unk_ix = 0
            else:
                self._unk_ix += 1
            return self._n_vecs + self.n_placeholder_rot*2 + self._unk_ix

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        word_to_vec = loader.load_word_vec(self.vec_name)
        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]

        null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
        matrix_list = [null_embed]

        ix = len(matrix_list)

        mat = []
        for word in voc:
            if word in self._word_to_ix:
                pass
            # elif is_number(word):
            #     self._word_to_ix[word] = -1
            elif self.is_named(word):
                self._word_to_ix[word] = -2
            else:
                word = word.lower()
                if word not in self._word_to_ix:
                    if word in word_to_vec:
                        mat.append(word_to_vec[word])
                        self._word_to_ix[word] = ix
                        ix += 1
                    else:
                        self._word_to_ix[word] = -3

        ids = np.array(list(self._word_to_ix.values()))
        print("Have %d num, %d named, %d unk , %d other" % (
            (ids == -1).sum(), (ids == -2).sum(), (ids == -3).sum(), (ids > 0).sum()))

        matrix_list.append(tf.constant(value=np.array(mat)))

        self._word_emb_mat = matrix_list
        self._n_vecs = ix

    def embed(self, is_train, *word_ix):
        dim = self._word_emb_mat[0].shape.as_list()[-1]
        if self.place_holder_scale == 0 or self.placeholder_dist is None:
            placeholders = [tf.zeros((self.n_placeholder_rot, dim), tf.float32) for _ in range(3)]
        elif self.placeholder_dist == "uniform":
            placeholders = [tf.random_uniform((self.n_placeholder_rot, dim),
                                                -self.place_holder_scale, self.place_holder_scale) for _ in range(3)]
        else:
            placeholders = [tf.random_normal((self.n_placeholder_rot, dim),
                                             0, self.place_holder_scale) for _ in range(3)]
        if self.placeholder_mapper:
            for i, name in enumerate(["num", "name", "unk"]):
                with tf.variable_scope("map_%s_placeholders" % name):
                    placeholders[i] = self.placeholder_mapper.apply(is_train, placeholders[i])

        mat = tf.concat(self._word_emb_mat + placeholders, axis=0)
        return [tf.nn.embedding_lookup(mat, x[0]) for x in word_ix]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)

    def __setstate__(self, state):
        super().__setstate__(state)


