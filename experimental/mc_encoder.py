from typing import List, Optional, Dict, Union

import tensorflow as tf
from tensorflow import Tensor

from configurable import Configurable
from data_processing.paragraph_qa import DocumentQaStats
from data_processing.qa_data import ParagraphAndQuestionDataset
from dataset import ListDataset, ListBatcher
from nn.embedder import WordEmbedder, CharWordEmbedder, CharEmbedder
import numpy as np


class McQuestion(object):
    def __init__(self, question_id: str, question: List[str],
                 answer_options: List[List[str]], context: List[List[str]], answer: int):
        self.question_id = question_id
        self.context = context
        self.question = question
        self.answer_options = answer_options
        self.answer = answer

    def get_text(self):
        voc = set()
        for sent in self.context:
            voc.update(sent)
        for ans in self.answer_options:
            voc.update(ans)
        voc.update(self.question)
        return voc


class McDataset(ListDataset):
    def __init__(self, data: List[McQuestion], batching: ListBatcher, unfiltered_len: Optional[int] = None):
        super().__init__(data, batching, unfiltered_len)

    def get_vocab(self):
        voc = set()
        for point in self.data:
            for sent in point.context:
                voc.update(sent)
            for ans in point.answer_options:
                voc.update(ans)
            voc.update(point.question)
        return voc


class McQuestionEncoder(Configurable):
    def __init__(self,
                 para_size_th: Optional[int]=None,
                 sent_size_th: Optional[int]=None):
        # Parameters
        self.doc_size_th = para_size_th
        self.sent_size_th = sent_size_th

        self._word_embedder = None
        self._char_emb = None

        # Internal stuff we need to set on `init`
        self.batch_size = None
        self.n_options = None
        self.max_context_word_dim = None
        self.max_ques_word_dim = None
        self.max_char_dim = None

        self.context_words = None
        self.context_chars = None
        self.context_len = None

        self.question_words = None
        self.question_chars = None
        self.question_len = None

        self.answer_words = None
        self.answer_chars = None
        self.answer_len = None

        self.answer = None

    @property
    def version(self):
        # version 1: added word_featurizer
        # version 2: answer encoder is now modular
        return 2

    def init(self, batch_size: int, n_options: int,
             word_emb: WordEmbedder, char_emb: Optional[CharEmbedder]):
        self.n_options = n_options
        self._word_embedder = word_emb
        self._char_emb = char_emb

        self.batch_size = batch_size

        if self._char_emb is not None:
            self.max_char_dim = self._char_emb.get_word_size_th()
        else:
            self.max_char_dim = 1

        batch_size = self.batch_size

        self.answer = tf.placeholder('int32', [batch_size], name='answer')

        self.answer_words = tf.placeholder('int32', [batch_size, n_options, None], name='answer_words')
        self.answer_len = tf.placeholder('int32', [batch_size, n_options], name='answer_len')

        self.question_words = tf.placeholder('int32', [batch_size, None], name='question_words')
        self.question_len = tf.placeholder('int32', [batch_size], name='question_len')

        self.context_words = tf.placeholder('int32', [batch_size, None], name='context_words')
        self.context_len = tf.placeholder('int32', [batch_size], name='context_len')

        if self._char_emb:
            self.context_chars = tf.placeholder('int32', [batch_size, None, self.max_char_dim], name='context_chars')
            self.question_chars = tf.placeholder('int32', [batch_size, None, self.max_char_dim], name='question_chars')
            self.answer_chars = tf.placeholder('int32', [batch_size, n_options, None, self.max_char_dim], name='question_chars')
        else:
            self.context_chars = None
            self.question_chars = None
            self.answer_chars = None

    def get_placeholders(self):
        return [x for x in
                [self.question_len, self.question_words, self.question_chars,
                 self.answer_len, self.answer_words, self.answer_chars,
                 self.context_len, self.context_words, self.context_chars,
                 self.answer]
                if x is not None]

    def encode(self, batch: List[McQuestion], is_train: bool):
        batch_size = len(batch)
        if self.batch_size is not None:
            if self.batch_size < batch_size:
                raise ValueError("Batch sized we pre-specified as %d, but got a batch of %d" % (self.batch_size, batch_size))
            # We have a fixed batch size, so we will pad our inputs with zeros along the batch dimension
            batch_size = self.batch_size

        feed_dict = {}
        max_char_dim = self.max_char_dim

        context_len = np.array([sum(len(s) for s in doc.context) for doc in batch], dtype='int32')
        question_len = np.array([len(x.question) for x in batch], dtype='int32')
        answer_len = np.array([[len(a) for a in x.answer_options] for x in batch], dtype='int32')

        feed_dict[self.context_len] = context_len
        feed_dict[self.question_len] = question_len
        feed_dict[self.answer_len] = answer_len

        ques_word_dim = question_len.max()
        context_word_dim = context_len.max()
        answer_word_dim = answer_len.max()

        context_words = np.zeros([batch_size, context_word_dim], dtype='int32')
        question_words = np.zeros([batch_size, ques_word_dim], dtype='int32')
        answer_words = np.zeros([batch_size, self.n_options, answer_word_dim], dtype='int32')
        feed_dict[self.context_words] = context_words
        feed_dict[self.question_words] = question_words
        feed_dict[self.answer_words] = answer_words

        if self._char_emb is not None:
            context_chars = np.zeros([batch_size, context_word_dim, max_char_dim], dtype='int32')
            question_chars = np.zeros([batch_size, ques_word_dim, max_char_dim], dtype='int32')
            answer_chars = np.zeros([batch_size, self.n_options, answer_word_dim, max_char_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
            feed_dict[self.context_chars] = context_chars
            feed_dict[self.answer_chars] = answer_chars
        else:
            context_chars, question_chars, answer_chars = None, None, None

        query_once = self._word_embedder.query_once()

        for doc_ix, doc in enumerate(batch):
            doc_mapping = {}

            for word_ix, word in enumerate(doc.question):
                if query_once:
                    ix = doc_mapping.get(word)
                    if ix is None:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                        doc_mapping[word] = ix
                else:
                    ix = self._word_embedder.context_word_to_ix(word, is_train)
                question_words[doc_ix, word_ix] = ix
                if self._char_emb is not None:
                    for char_ix, char in enumerate(word):
                        if char_ix == self.max_char_dim:
                            break
                        question_chars[doc_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

            word_ix = 0
            for sent_ix, sent in enumerate(doc.context):
                if self.sent_size_th is not None and sent_ix == self.sent_size_th:
                    break
                for word in sent:
                    if word_ix == self.max_context_word_dim:
                        break
                    if query_once:
                        ix = doc_mapping.get(word)
                        if ix is None:
                            ix = self._word_embedder.context_word_to_ix(word, is_train)
                            doc_mapping[word] = ix
                    else:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                    context_words[doc_ix, word_ix] = ix

                    if self._char_emb is not None:
                        for char_ix, char in enumerate(word):
                            if char_ix == self.max_char_dim:
                                break
                            context_chars[doc_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)
                    word_ix += 1

            for answer_ix, answer in enumerate(doc.answer_options):
                for word_ix, word in enumerate(answer):
                    if query_once:
                        ix = doc_mapping.get(word)
                        if ix is None:
                            ix = self._word_embedder.context_word_to_ix(word, is_train)
                            doc_mapping[word] = ix
                    else:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                    question_words[doc_ix, answer_ix, word_ix] = ix
                    if self._char_emb is not None:
                        for char_ix, char in enumerate(word):
                            if char_ix == self.max_char_dim:
                                break
                            question_chars[doc_ix, answer_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

        feed_dict[self.answer] = np.array([x.answer for x in batch], dtype=np.int32)
        return feed_dict
