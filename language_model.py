import sys
from typing import List, Optional, Set

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configurable import Configurable
from data_processing.batching import get_batches, get_clustered_batches
from data_processing.paragraph_qa import ParagraphQaCollection
from dataset import Dataset, TrainingData
from model import Model, ModelOutput
from nn.embedder import WordEmbedder
from nn.layers import SequenceMapper
from utils import ResourceLoader, flatten_iterable


class LanguageModelEncoder(object):

    def __init__(self, max_context_word_dim):
        self._word_embedder = None
        self.batch_size = None
        self.words = None
        self.words_len = None
        self.max_context_word_dim = max_context_word_dim

    def init(self, batch_size: Optional[int], word_emb: WordEmbedder):
        self._word_embedder = word_emb
        self.batch_size = batch_size
        self.words = tf.placeholder('int32', [self.batch_size, None], name='context_words')
        self.words_len = tf.placeholder('int32', [self.batch_size], name='context_len')

    def get_placeholders(self):
        return [self.words, self.words_len]

    def encode(self, batch: List, is_train: bool):
        batch_size = len(batch)
        if self.batch_size is not None:
            if self.batch_size < batch_size:
                raise ValueError()

        word_len = np.array([sum(len(s) for s in para) for para in batch], dtype='int32')
        if self.max_context_word_dim is not None:
            word_len = np.min(word_len, self.max_context_word_dim)
        words = np.zeros([batch_size, word_len.max()], dtype='int32')

        word_ix = 0
        for para_ix, para in enumerate(batch):
            for sent_ix, sent in enumerate(para):
                for word in sent:
                    if self.max_context_word_dim is not None and word_ix == self.max_context_word_dim:
                        break
                    ix = self._word_embedder.question_word_to_ix(word)
                    words[para_ix, word_ix] = ix

        return dict(words=words, word_len=word_len)


class LmBatchingParameters(Configurable):
    def __init__(self,
                 train_batch_size: int,
                 test_batch_size: int,
                 train_cluster: Optional[int],
                 eval_cluster: Optional[int],
                 shuffle_buckets=True,
                 truncate_batches=False):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_cluster = train_cluster
        self.eval_cluster = eval_cluster
        self.shuffle_buckets = shuffle_buckets
        self.truncate_batches = truncate_batches

    def get_batches(self, n_epochs, n_examples, data: List, is_train):
        # TODO api should really support `n_examples` over n_batches
        batch_size = self.train_batch_size if is_train else self.test_batch_size
        n_batches = 0 if n_examples is None else n_examples// batch_size
        cluster = self.train_cluster if is_train else self.eval_cluster
        cluster_fn = lambda x: len(x) / cluster
        if cluster_fn is None:
            return get_batches(data, batch_size, n_epochs, n_batches,
                               allow_truncate=self.truncate_batches)
        else:
            return get_clustered_batches(data, batch_size, cluster_fn, n_epochs, n_batches,
                                         shuffle_buckets=self.shuffle_buckets and is_train,
                                         allow_truncate=self.truncate_batches)


class LanguageTrainingData(TrainingData):
    def __init__(self, batching: LmBatchingParameters, corpus, seed=0,
                 min_doc_len=10, intern=False,
                 upper_start=True,
                 percent_dev=0.1, sample_documents=None):
        self.upper_start = upper_start
        self.min_para_len = min_doc_len
        self.seed = seed
        self.intern = intern
        self.percent_dev = percent_dev
        self.sample_documents = sample_documents
        self.batching = batching
        self.corpus = corpus
        self._train = None
        self._eval = None

    def _load(self):
        print("Scanning files...")
        filenames = self.corpus.list_documents()
        filenames.sort()
        np.random.RandomState(self.seed).shuffle(filenames)
        if self.sample_documents:
            filenames = filenames[:self.sample_documents]
        print("Loading docs...")
        docs = []
        for filename in tqdm(filenames):
            doc = self.corpus.get_document(filename, flat=False)
            if doc is None:
                raise ValueError(filename)
            paras = []
            for para in doc:
                n_words = sum(len(s) for s in para)
                if n_words < self.min_para_len:
                    continue
                first_word = para[0][0]
                if self.upper_start and not first_word[0].isupper():
                    continue
                p = []
                for sent in para:
                    if self.intern:
                        p += [sys.intern(x) for x in sent]
                    else:
                        p += sent
                paras.append(p)
            docs.append(paras)

        print("Done")
        n_eval = int(len(docs) * self.percent_dev)
        self._train = flatten_iterable([x for x in docs[n_eval:]])
        self._eval = flatten_iterable([x for x in docs[:n_eval]])

    def get_train(self) -> Dataset:
        self._load()
        return LanguageDataset(self._train, self.batching)

    def get_train_corpus(self) -> object:
        return None

    def get_eval(self) -> Dataset:
        self._load()
        return LanguageDataset(self._eval, self.batching)

    def get_resource_loader(self) -> ResourceLoader:
        return ResourceLoader()


class LanguageDataset(Dataset):
    def __init__(self, data: List[List[List[str]]], batching: LmBatchingParameters):
        self.data = data
        self.batching = batching

    def get_batches(self, n_epochs: int, n_elements: int = 0, is_train: bool = True):
        return self.batching.get_batches(n_epochs, n_elements, self.data, is_train)

    def get_spec(self) -> Optional[int]:
        if self.batching.truncate_batches:
            return None
        else:
            return max(self.batching.test_batch_size, self.batching.train_batch_size)

    def get_vocab(self) -> Set[str]:
        voc = set()
        for para in self.data:
            for sent in para:
                voc.update(sent)
        return voc

    def __len__(self):
        return len(self.data)


class LanguageModelBase(Model):
    def __init__(self, encoder: LanguageModelEncoder, embedder: WordEmbedder):
        self.encoder = encoder
        self.embedder = embedder
        self._is_train_placeholder = None

    def init(self, corpus: ParagraphQaCollection, loader: ResourceLoader):
        self.embedder.set_vocab(corpus, loader, corpus.special_tokens)

    def set_inputs(self, datasets: List[LanguageDataset], word_vec_loader):
        self._is_train_placeholder = tf.placeholder(tf.bool, ())
        voc = datasets[0].get_vocab()
        batch_size = datasets[0].get_spec()
        for dataset in datasets[1:]:
            voc.update(dataset.get_vocab())
            if batch_size is not None:
                other_batch_size = dataset.get_spec()
                if other_batch_size is not None:
                    batch_size = max(batch_size, other_batch_size)
                else:
                    batch_size = None

        self.set_input_spec(batch_size, voc, word_vec_loader)

    def set_input_spec(self, batch_size, voc, word_vec_loader):
        if self.embedder is not None:
            self.embedder.init(word_vec_loader, voc)
        self.encoder.init(batch_size, self.embedder)

    def get_prediction(self) -> ModelOutput:
        return self.get_predictions_for(self._is_train_placeholder, {x: x for x in self.encoder.get_placeholders()})

    def get_predictions_for(self, is_train: tf.Tensor, input_tensors) -> ModelOutput:
        enc = self.encoder
        mask = input_tensors[enc.words_len]

        with tf.variable_scope("word-embed"):
            words = self.embedder.embed(is_train, (input_tensors[enc.words], mask))[0]
        return self._get_predictions_for(is_train, words. mask)

    def _get_predictions_for(self, is_train, words, mask):
        raise NotImplementedError()

    def encode(self, batch, is_train: bool):
        data = self.encoder.encode(batch, is_train)
        data[self._is_train_placeholder] = is_train
        return data


class PartialFillIn(LanguageModelBase):
    def __init__(self, encoder: LanguageModelEncoder, embedder: WordEmbedder,
                 mapper: SequenceMapper, fill_in_percent: float, loss_fn="l1"):
        super().__init__(encoder, embedder)
        self.mapper = mapper
        self.fill_in_percent = fill_in_percent
        self.loss_fn = loss_fn

    def _get_predictions_for(self, is_train, words, mask):
        dropped_out = tf.nn.dropout(words, self.fill_in_percent)
        predictions = self.mapper.apply(is_train, dropped_out, mask)
        if self.loss_fn == "l1":
            loss = tf.reduce_mean(tf.abs(predictions - words))
        elif self.loss_fn == "l2":
            loss = tf.reduce_mean(tf.square(predictions - words))
        else:
            raise ValueError()
        return ModelOutput(loss, predictions)

