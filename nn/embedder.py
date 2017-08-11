from collections import Counter
from typing import List, Iterable, Optional

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

from configurable import Configurable
from data_processing.paragraph_qa import ParagraphAndQuestion
from nn.layers import SqueezeLayer, Updater
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

    def question_word_to_ix(self, word):
        raise NotImplementedError()

    def context_word_to_ix(self, word):
        raise NotImplementedError()

    def get_placeholder(self, ix, is_train):
        """
        Words given a negative index by `question_word_to_ix` or `context_word_to_ix` will get assigned a new
        index as produced by `get_placeholder`, where all identical words will get the same placeholder
        """
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

    def __init__(self, word_size_th: int, char_th: int, char_dim: int, init_scale: float=0.1):
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

    def init(self, word_vec_loader, voc: Iterable[str]):
        pass

    def char_to_ix(self, char):
        return self._char_to_ix.get(char, 1)

    def embed(self, is_train, *char_ix):
        zero = tf.zeros((1, self.char_dim), dtype=np.float32)
        mat = tf.get_variable("char_emb_mat", (len(self._char_to_ix)+1, self.char_dim),
                              tf.float32, initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
        emb_mat = tf.concat([zero, mat], axis=0)

        x = char_ix[0]
        tf.nn.embedding_lookup(emb_mat, x[0])

        return [tf.nn.embedding_lookup(emb_mat, x[0]) for x in char_ix]


class CharWordEmbedder(Configurable):
    """
    Derives word embeddings from character embeddings by combining a character embedder and a reduce layer
    """
    def __init__(self, embedder: CharEmbedder, layer: SqueezeLayer, shared_parameters: bool):
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


class FixedWordEmbedder(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0.05,
                 learn_unk: bool = True,
                 keep_probs: float = 1,
                 noise_imp = "unique",
                 cpu=False):
        self.dropout_imp = noise_imp
        self.keep_probs = keep_probs
        self.word_vec_init_scale = word_vec_init_scale
        self.learn_unk = learn_unk
        self.vec_name = vec_name
        self.cpu = cpu

        # Built in `init`
        self._word_to_ix = None
        self._word_emb_mat = None
        self._special_tokens = None

    def set_vocab(self, data: List[ParagraphAndQuestion], loader: ResourceLoader, special_tokens: List[str]):
        if special_tokens is not None:
            self._special_tokens = sorted(special_tokens)

    def is_vocab_set(self):
        return True

    def question_word_to_ix(self, word):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def context_word_to_ix(self, word):
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

        null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
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
        if self.keep_probs < 1:
            # TODO there is probably a more efficient way to do this....
            # This only one I can think of is to keep a variable around of the dropped-out matrix
            # and use tf.unique adn tf.scatter_update to "refersh" the dropouts for the correct
            # indices each iteration
            mat = tf.cond(is_train,
                          lambda: tf.nn.dropout(mat, self.keep_probs),
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

        # Words/Chars we will embed
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


class ShuffleNames(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0,
                 keep_probs: float = 1,
                 place_holder_scale = 0.05,
                 drop_named: bool = False,
                 learn_unk: bool = False,
                 named_thresh=0.9):
        self.drop_named = drop_named
        self.keep_probs = keep_probs
        self.learn_unk = learn_unk
        self.word_vec_init_scale = word_vec_init_scale
        self.vec_name = vec_name
        self.place_holder_scale = place_holder_scale
        self.named_thresh = named_thresh

        # corpus stats we need to keep around
        self._word_counts = None
        self._word_counts_lower = None
        self._stop = None

        self._word_to_ix = None
        self._word_emb_mat = None
        self._name_start = 0
        self._name_end = 0

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
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def context_word_to_ix(self, word):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def get_placeholder(self, ix, is_train):
        return np.random.randint(self._name_start, self._name_end)

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        word_to_vec = loader.load_word_vec(self.vec_name)
        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]

        null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
        unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
                                    dtype=np.float32, trainable=self.learn_unk,
                                    initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                              self.word_vec_init_scale))
        matrix_list = [null_embed, unk_embed]
        ix = len(matrix_list)

        mat = []
        name_mat = []
        for word in voc:
            if word in self._word_to_ix:
                continue  # in case we already added due after seeing a capitalized version of `word`
            named = self.is_named(word)
            if word in word_to_vec:
                if named:
                    name_mat.append(word_to_vec[word])
                    self._word_to_ix[word] = -1
                else:
                    mat.append(word_to_vec[word])
                    self._word_to_ix[word] = ix
                    ix += 1
            else:
                lower = word.lower()  # Full back to the lower-case version
                if lower in word_to_vec and lower not in self._word_to_ix:
                    if named:
                        name_mat.append(word_to_vec[lower])
                        self._word_to_ix[word] = -1
                    else:
                        mat.append(word_to_vec[lower])
                        self._word_to_ix[lower] = ix
                        ix += 1
                elif named:
                    self._word_to_ix[word] = -1

        ids = np.array(list(self._word_to_ix.values()))
        print("Have %d named (and %d name vecs), %d other" % (
            (ids == -1).sum(), len(name_mat),(ids > 1).sum()))

        if self.drop_named:
            for k, i in self._word_to_ix.items():
                if i == -1:
                    self._word_to_ix[k] = 1
            matrix_list.append(tf.constant(value=np.array(mat)))
        else:
            matrix_list.append(tf.constant(value=np.concatenate([np.array(mat), np.array(name_mat)], axis=0)))
        self._name_start = ix
        self._name_end = ix + len(name_mat)
        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def embed(self, is_train, *word_ix):
        return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)

    def __setstate__(self, state):
        super().__setstate__(state)


class DropShuffleNames(WordEmbedder):

    def __init__(self,
                 vec_name: str,
                 word_vec_init_scale: float = 0,
                 name_keep_probs: float=0.5,
                 learn_unk: bool = False,
                 named_thresh=0.9):
        self.name_keep_probs = name_keep_probs
        self.learn_unk = learn_unk
        self.word_vec_init_scale = word_vec_init_scale
        self.vec_name = vec_name
        self.named_thresh = named_thresh

        # corpus stats we need to keep around
        self._word_counts = None
        self._word_counts_lower = None
        self._stop = None
        self._name_vecs_start = 0
        self._name_vecs_end = 0

        self._word_to_ix = None
        self._word_emb_mat = None

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
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def context_word_to_ix(self, word):
        ix = self._word_to_ix.get(word, 1)
        if ix == 1:
            return self._word_to_ix.get(word.lower(), 1)
        else:
            return ix

    def init(self, loader: ResourceLoader, voc: Iterable[str]):
        word_to_vec = loader.load_word_vec(self.vec_name)
        self._word_to_ix = {}

        dim = next(iter(word_to_vec.values())).shape[0]

        null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
        unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
                                    dtype=np.float32, trainable=self.learn_unk,
                                    initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                              self.word_vec_init_scale))
        matrix_list = [null_embed, unk_embed]
        ix = len(matrix_list)

        mat = []
        names = []
        for word in voc:
            if word in self._word_to_ix:
                continue  # in case we already added due after seeing a capitalized version of `word`
            if self.is_named(word):
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
                                   dtype=np.float32,  initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
                                                                              self.word_vec_init_scale))
        matrix_list.append(name_unk)
        ix += 1
        name_mat = []
        for name in names:
            vec = word_to_vec.get(name)
            if vec is not None:
                name_mat.append(vec)
                self._word_to_ix[name] = -ix
                ix += 1
            else:
                word_to_vec[name] = self._name_vecs_start  # unk for names
        matrix_list.append(tf.constant(np.array(name_mat)))
        self._name_vecs_end = ix

        print("Have %d named (and %d name vecs), %d other" % (
            len(names), len(name_mat), len(mat)))

        self._word_emb_mat = tf.concat(matrix_list, axis=0)

    def get_placeholder(self, ix, is_train):
        if not is_train or np.random.random() < self.name_keep_probs:
            return abs(ix)
        return np.random.randint(self._name_vecs_start, self._name_vecs_end)

    def embed(self, is_train, *word_ix):
        return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_word_emb_mat"] = None  # we will rebuild these anyway
        state["_word_to_ix"] = None
        return dict(version=self.version, state=state)

    def __setstate__(self, state):
        super().__setstate__(state)


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

# class ScatterPlaceholders(WordEmbedder):
#     """
#     Simple placeholder strategy: keep a matrix of random vectors and
#     randomly assign unknown words to one of those vectors
#     """
#
#     def __init__(self,
#                  vec_name: str,
#                  placeholder_scale: Optional[float],
#                  placeholder_dist: str,
#                  placeholder_map: Optional[Updater],
#                  n_placeholders: int,
#                  learn_placeholders: bool=False,
#                  ):
#         self.placeholder_dist = placeholder_dist
#         self.learn_placeholders = learn_placeholders
#         self.placeholder_scale = placeholder_scale
#         self.n_placeholders = n_placeholders
#         self.placeholder_map = placeholder_map
#
#         self.vec_name = vec_name
#
#         # Built in `init`
#         self._word_mat = None
#         self._place_holder_mat = None
#         self._on_place_holder = None
#         self._place_holder_ix = None
#
#         self._word_to_ix = None
#         self._word_emb_mat = None
#
#
#
#     def is_vocab_set(self):
#         return True
#
#     def placeholder_ix(self):
#         return -1
#
#     def question_word_to_ix(self, word):
#         return self._word_to_ix.get(word.lower(), -1)
#
#     def context_word_to_ix(self, word):
#         return self._word_to_ix.get(word.lower(), -1)
#
#     @property
#     def word_embed_mat(self):
#         return self._word_emb_mat
#
#     def init(self, word_vec_loader, voc: Iterable[str]):
#         word_to_vec = word_vec_loader.load_word_vec(self.vec_name)
#         dim = next(iter(word_to_vec.values())).shape[0]
#
#         self._word_to_ix = {}
#         ix = 1
#
#         mat = [np.zeros(dim, dtype=np.float32)]
#         for word in voc:
#             word = word.lower()
#             if word in word_to_vec and word not in self._word_to_ix:
#                 mat.append(word_to_vec[word])
#                 self._word_to_ix[word] = ix
#                 ix += 1
#
#         mat = np.vstack(mat)
#         self._word_mat = tf.constant(mat)
#
#         scale = self.placeholder_scale
#
#         if self.placeholder_dist == "uniform":
#             if scale is None:
#                 scale = np.mean(np.abs(mat))
#             place_init = tf.random_uniform_initializer(minval=-scale, maxval=scale)
#         elif self.placeholder_dist == "normal":
#             if scale is None:
#                 scale = np.std(mat)
#             place_init = tf.random_normal_initializer(stddev=scale)
#         else:
#             raise ValueError()
#         place_holder_mat = tf.get_variable("placeholders", (self.n_placeholders, dim), dtype=tf.float32,
#                                            initializer=place_init, trainable=self.learn_placeholders)
#         self._place_holder_mat = place_holder_mat
#         self._on_place_holder = 0
#         self._place_holder_ix = np.arange(ix, ix+self.n_placeholders)
#         np.random.shuffle(self._place_holder_ix)
#
#     def embed(self, is_train, *word_ix):
#         if self.placeholder_map is not None:
#             placeholder = self.placeholder_map.apply(is_train, self._place_holder_mat)
#         else:
#             placeholder = self._place_holder_mat
#
#         word_emb_mat = tf.concat([self._word_mat, placeholder], axis=0)
#
#         if any(len(x) != 2 for x in word_ix):
#             raise ValueError()
#         return [tf.nn.embedding_lookup(word_emb_mat, x[0]) for x in word_ix]


# class PartialTrainEmbedder(WordEmbedder, Configurable):
#
#     def __init__(self,
#                  vec_name: str,
#                  word_count_th: int = 50,
#                  para_size_th: int = 400,
#                  word_vec_init_scale: float = 0.05,
#                  learn_unk: bool = True,
#                  n_untrained: int = 1,
#                  embed_keep: float = 1,
#                  train_context_th = ((2100, 80), (2100, 80)),
#                  train_question_th = ((2000, 0),),
#                  seperate_question: bool = True,
#                  train_unknown: bool = True):
#         self.embed_keep = embed_keep
#         self.n_untrained = n_untrained
#         self.word_vec_init_scale = word_vec_init_scale
#         self.learn_unk = learn_unk
#         self.seperate_question = seperate_question
#
#         self.train_question_th = train_question_th
#         self.train_unknown = train_unknown
#         self.train_context_th = train_context_th
#
#         self.vec_name = vec_name
#         self.para_size_th = para_size_th
#         self.word_count_th = word_count_th
#
#         # Words/Chars we will embed
#         self.train_question_words = None
#         self.train_context_words = None
#
#         # Built in `init`
#         self._context_word_to_ix = None
#         self._question_word_to_ix = None
#
#         self._char_to_ix = None
#         self._word_emb_mat = None
#         self._char_emb_mat = None
#
#     def set_vocab(self, data: List[ParagraphAndQuestion], word_vec_loader):
#         paras = {}
#         context_word_counter = Counter()
#         context_article_counter = Counter()
#         question_word_counter = Counter()
#         question_article_counter = Counter()
#         article_question_words = defaultdict(set)
#
#         for data_point in data:
#             paras[(data_point.article_id, data_point.paragraph_num)] = data_point.context
#             question_words = set()
#             for word in data_point.question:
#                 question_word_counter[word.lower()] += 1
#                 question_words.add(word.lower())
#             article_question_words[data_point.paragraph_num].update(question_words)
#         for v in article_question_words.values():
#             question_article_counter.update(v)
#
#         for para in paras.values():
#             article_set = set()
#             for sent in para:
#                 for word in sent:
#                     if word[0].islower():
#                         context_word_counter[word.lower()] += 1
#                         article_set.add(word.lower())
#             for word in article_set:
#                 context_article_counter[word] += 1
#
#         combined_counters = context_word_counter + question_word_counter
#
#         vecs = word_vec_loader.get_word_vecs(self.vec_name)
#
#         self.train_context_words = []
#         self.train_question_words = []
#
#         for word, count in combined_counters.items():
#             if count >= self.word_count_th and (self.train_unknown or word in vecs):
#                 article_count = context_article_counter[word]
#                 context_count = context_word_counter[word]
#                 question_count = question_word_counter[word]
#                 question_article_count = question_article_counter[word]
#                 if any(context_count >= word_c and article_count >= art_c
#                        for word_c, art_c in self.train_context_th):
#                     self.train_context_words.append(word)
#                 if any(question_count >= word_c and question_article_count >= art_c
#                        for word_c, art_c in self.train_question_th):
#                     self.train_question_words.append(word)
#
#         if not self.seperate_question:
#             self.train_context_words = set(self.train_context_words)
#             self.train_context_words.update(self.train_question_words)
#             self.train_context_words = list(self.train_context_words)
#             self.train_question_words = []
#
#         print("Training %d context words and %d question words" % (
#             len(self.train_context_words), len(self.train_question_words)))
#
#     def is_vocab_set(self):
#         return self.train_context_words is not None
#
#     def question_word_to_ix(self, word):
#         return self._question_word_to_ix.get(word.lower(), 1)
#
#     def context_word_to_ix(self, word):
#         return self._context_word_to_ix.get(word.lower(), 1)
#
#     def init(self, word_vec_loader, voc: Iterable[str]):
#         word_to_vec = word_vec_loader.get_word_vecs(self.vec_name)
#         self._context_word_to_ix = {}
#         self._question_word_to_ix = {}
#         ix = 2
#
#         dim = next(iter(word_to_vec.values())).shape[0] + self.n_untrained
#
#         null_embed = tf.constant(np.zeros((1, dim), dtype=np.float32))
#         unk_embed = tf.get_variable(shape=(1, dim), name="unk_embed",
#                                     dtype=np.float32, trainable=self.learn_unk,
#                                     initializer=tf.random_uniform_initializer(-self.word_vec_init_scale, self.word_vec_init_scale))
#         matrix_list = [null_embed, unk_embed]
#
#         if len(self.train_context_words) > 0:
#             train_embed = tf.get_variable("train_context_emb_mat",
#                                           shape=[len(self.train_context_words), dim], dtype='float32', trainable=True,
#                                           initializer=lambda s, **kwargs: self._init_word_vec(self.train_context_words, word_to_vec))
#             for word in self.train_context_words:
#                 self._context_word_to_ix[word] = ix
#                 if not self.seperate_question:
#                     self._question_word_to_ix[word] = ix
#                 ix += 1
#
#             matrix_list.append(train_embed)
#
#         if len(self.train_question_words) > 0:
#             train_embed = tf.get_variable("train_question_emb_mat",
#                                           shape=[len(self.train_question_words), dim], dtype='float32', trainable=True,
#                                           initializer=lambda s, **kwargs: self._init_word_vec(self.train_question_words, word_to_vec))
#             for word in self.train_question_words:
#                 self._question_word_to_ix[word] = ix
#                 ix += 1
#             matrix_list.append(train_embed)
#
#         voc = [w.lower() for w in voc]
#         voc = [w for w in voc if (w not in self._question_word_to_ix or w not in self._question_word_to_ix) and w in word_to_vec]
#
#         fixed_embed = tf.get_variable("fixed_emb_mat",
#                                       shape=[len(voc), dim], dtype='float32',
#                                       initializer=lambda s, **kwargs: self._init_word_vec(voc, word_to_vec))
#         matrix_list.append(fixed_embed)
#
#         for word in voc:
#             if word not in self._question_word_to_ix:
#                 self._question_word_to_ix[word] = ix
#             if word not in self._context_word_to_ix:
#                 self._context_word_to_ix[word] = ix
#             ix += 1
#
#         self._word_emb_mat = tf.concat(matrix_list, axis=0)
#
#     def _init_word_vec(self, words, word2vec_dict):
#         vec_dim = next(iter(word2vec_dict.values())).shape[0]
#         dim = vec_dim + self.n_untrained
#         scale = self.word_vec_init_scale
#
#         matrix = np.zeros((len(words), dim), dtype=np.float32)
#         for ix, word in enumerate(words):
#             if word in word2vec_dict:
#                 vec = word2vec_dict[word]
#                 padded_vec = np.empty(dim, np.float32)
#                 padded_vec[:vec_dim] = vec
#                 padded_vec[vec_dim:] = 0
#                 matrix[ix] = padded_vec
#             else:
#                 matrix[ix] = np.random.uniform(low=-scale, high=scale, size=dim)
#
#         return matrix
#
#     @property
#     def word_embed_mat(self):
#         return self._word_emb_mat

### The following embedder are pretty experimental, so be careful...

# class SelectiveDropout(WordEmbedder):
#
#     def __init__(self,
#                  vec_name: str,
#                  learn_unk: bool = True,
#                  word_vec_init_scale: float = 0,
#                  rare_word_th: int = 5,
#                  keep_probs: float = 1,
#                  noise_imp = "unique",
#                  cpu=False):
#         self.rare_word_th = rare_word_th
#         self.dropout_imp = noise_imp
#         self.keep_probs = keep_probs
#         self.word_vec_init_scale = word_vec_init_scale
#         self.learn_unk = learn_unk
#         self.vec_name = vec_name
#         self.cpu = cpu
#
#         # Built in `init`
#         self._word_counts = None
#         self._word_counts_lower = None
#         self._word_to_ix = None
#         self._word_emb_mat = None
#         self._stop = None
#         self._top_n_minus_one = None
#
#     def set_vocab(self, corpus, loader: ResourceLoader):
#         # raise NotImplemented()
#         # for paras, questions in zip(corpus.paragraphs, corpus.questions):
#         #     article_counter = Counter()
#         #     for q in questions[article]:
#         #         article_counter.update(q)
#         #     for para in paras.values():
#         #         for sent in para:
#         #             article_counter.update(sent)
#         #     for k,v in article_counter.items():
#         #         article_word_counts[k].append(v)
#         #
#         # upper_counters = Counter()
#         # top_n_min_one = Counter()
#         # lower_counts = Counter()
#         #
#         # for k,v in article_word_counts.items():
#         #     total = sum(v)
#         #     upper_counters[k] = total
#         #     lower_counts[k.lower()] += total
#         #     top_n_min_one[k.lower()] += total - max(v)
#
#         # self._top_n_minus_one = top_n_min_one
#         # self._word_counts = upper_counters
#         # self._word_counts_lower = lower_counts
#         self._stop = set(stopwords.words('english'))
#
#     def is_named(self, word):
#         if word[0].isupper() and word[1:].islower():
#             wl = word.lower()
#             if wl not in self._stop:
#                 lc = self._word_counts_lower[wl]
#                 if lc == 0 or (self._word_counts[word] / lc) > 0.90:
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
#
#     @property
#     def word_embed_mat(self):
#         return self._word_emb_mat
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
#         named_embed = tf.get_variable(shape=(1, dim), name="named_embed",
#                                     dtype=np.float32, trainable=self.learn_unk,
#                                     initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
#                                                                               self.word_vec_init_scale))
#         num_embed = tf.get_variable(shape=(1, dim), name="num_embed",
#                                     dtype=np.float32, trainable=self.learn_unk,
#                                     initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
#                                                                               self.word_vec_init_scale))
#         rare_embed = tf.get_variable(shape=(1, dim), name="rare_embed",
#                                     dtype=np.float32, trainable=self.learn_unk,
#                                     initializer=tf.random_uniform_initializer(-self.word_vec_init_scale,
#                                                                               self.word_vec_init_scale))
#         matrix_list = [null_embed, unk_embed, num_embed, named_embed, rare_embed]
#
#         ix = len(matrix_list)
#
#         mat = []
#         for word in voc:
#             if word in self._word_to_ix:
#                 pass
#             elif is_number(word):
#                 self._word_to_ix[word] = 2
#             elif self.is_named(word):
#                 self._word_to_ix[word] = 3
#             elif self._top_n_minus_one[word.lower()] < self.rare_word_th:
#                 self._word_to_ix[word] = 4
#             else:
#                 word = word.lower()
#                 if word in word_to_vec and word not in self._word_to_ix:
#                     mat.append(word_to_vec[word])
#                     self._word_to_ix[word] = ix
#                     ix += 1
#
#         ids = np.array(list(self._word_to_ix.values()))
#         print("Have %d num, %d named, %d rare, %d other" % (
#             (ids == 2).sum(), (ids == 3).sum(), (ids == 4).sum(), (ids > 4).sum()
#         ))
#
#         matrix_list.append(tf.constant(value=np.array(mat)))
#
#         self._word_emb_mat = tf.concat(matrix_list, axis=0)
#
#     def reset(self):
#         self._word_emb_mat = None
#         self._word_to_ix = None
#
#     def embed(self, is_train, *word_ix):
#         if any(len(x) != 2 for x in word_ix):
#             raise ValueError()
#         mat = self._word_emb_mat
#         if self.keep_probs < 1:
#             # TODO there is probably a more efficient way to do this....
#             # This only one I can think of is to keep a variable around of the dropped-out matrix
#             # and use tf.unique adn tf.scatter_update to "refersh" the dropouts for the correct
#             # indices each iteration
#             mat = tf.cond(is_train,
#                           lambda: tf.nn.dropout(mat, self.keep_probs),
#                           lambda: mat)
#         if self.cpu:
#             with tf.device("/cpu:0"):
#                 return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
#         else:
#             return [tf.nn.embedding_lookup(self._word_emb_mat, x[0]) for x in word_ix]
#
