from typing import List, Optional

import numpy as np
import tensorflow as tf

from configurable import Configurable
from nn.embedder import WordEmbedder, CharEmbedder
from paragraph_selection.paragraph_selection_data import ParagraphQuestionGroup, \
    QuestionParagraphFeaturizer
from utils import transpose_lists


class ParagraphQuestionGroupEncoder(object):
    def __init__(self, truncate_paragraphs: Optional[int]=None):
        self.truncate_paragraphs = truncate_paragraphs

        self._word_embedder = None
        self._char_emb = None

        self.max_char_dim = None

        self.context_features = None
        self.context_sentences = None
        self.context_words = None
        self.context_chars = None
        self.context_len = None
        self.question_features = None
        self.question_words = None
        self.question_chars = None
        self.question_len = None
        self.answer_paragraph = None

    def get_placeholders(self):
        return [x for x in
                [self.question_len, self.question_words, self.question_chars, self.question_features,
                 self.context_len, self.context_words, self.context_chars, self.context_features,
                 self.answer_paragraph, self.context_sentences]
                if x is not None]

    def init(self, word_emb: WordEmbedder, char_emb: CharEmbedder):
        self._word_embedder = word_emb
        self._char_emb = char_emb

        if self._char_emb is not None:
            self.max_char_dim = self._char_emb.get_word_size_th()
        else:
            self.max_char_dim = 1

        self.context_sentences = tf.placeholder('int32', [None, None])
        self.context_words = tf.placeholder('int32', [None, None], name='context_words')
        self.context_len = tf.placeholder('int32', [None], name='context_len')

        self.question_words = tf.placeholder('int32', [None, None], name='question_words')
        self.question_len = tf.placeholder('int32', [None], name='question_len')

        if self._char_emb:
            self.context_chars = tf.placeholder('int32', [None, None, self.max_char_dim], name='context_chars')
            self.question_chars = tf.placeholder('int32', [None, None, self.max_char_dim], name='question_chars')
        else:
            self.context_chars = None
            self.question_chars = None

        self.answer_paragraph = tf.placeholder('int32', [None], name='answer_start')

    def encode(self, batch: ParagraphQuestionGroup, is_train: bool):
        max_char_dim = self.max_char_dim
        n_questions = len(batch.questions)
        n_paragraphs = len(batch.paragraphs)

        if len(batch.answer) != n_questions:
            raise ValueError()

        feed_dict = {}

        context_len = np.array([sum(len(s) for s in para) for para in batch.paragraphs], dtype='int32')
        question_len = np.array([len(q) for q in batch.questions], dtype='int32')

        feed_dict[self.context_len] = context_len
        feed_dict[self.question_len] = question_len

        context_sent_dim = max(len(para) for para in batch.paragraphs)
        context_sentences = np.zeros((n_paragraphs, context_sent_dim), dtype=np.int32)
        for ix, para in enumerate(batch.paragraphs):
            for sent_ix, sent in enumerate(para):
                context_sentences[ix, sent_ix] = len(sent)
        feed_dict[self.context_sentences] = context_sentences

        ques_word_dim = question_len.max()
        context_word_dim = context_len.max()
        if self.truncate_paragraphs is not None:
            context_word_dim = min(context_word_dim, self.truncate_paragraphs)

        if self._word_embedder is not None:
            context_words = np.zeros([n_paragraphs, context_word_dim], dtype='int32')
            question_words = np.zeros([n_questions, ques_word_dim], dtype='int32')
            feed_dict[self.context_words] = context_words
            feed_dict[self.question_words] = question_words
        else:
            question_words, context_words = None, None

        if self._char_emb is not None:
            context_chars = np.zeros([n_paragraphs, context_word_dim, max_char_dim], dtype='int32')
            question_chars = np.zeros([n_questions, ques_word_dim, max_char_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
            feed_dict[self.context_chars] = context_chars
        else:
            context_chars, question_chars = None, None

        placeholders = {}
        for para_ix, para in enumerate(batch.paragraphs):
            word_ix = 0
            for sent_ix, sent in enumerate(para):
                for word in sent:
                    if word_ix == self.truncate_paragraphs:
                        break
                    if self._word_embedder is not None:
                        ix = self._word_embedder.context_word_to_ix(word)
                        if ix < 0:
                            wl = word.lower()
                            if wl in placeholders:
                                ix = placeholders[wl]
                            else:
                                ix = self._word_embedder.get_placeholder(ix)
                                placeholders[wl] = ix

                        context_words[para_ix, word_ix] = ix

                    if self._char_emb is not None:
                        for char_ix, char in enumerate(word):
                            if char_ix == self.max_char_dim:
                                break
                            context_chars[para_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)
                    word_ix += 1

        for question_ix, question in enumerate(batch.questions):
            for word_ix, word in enumerate(question):
                if self._word_embedder is not None:
                    ix = self._word_embedder.question_word_to_ix(word)
                    if ix < 0:
                        wl = word.lower()
                        if wl in placeholders:
                            ix = placeholders[wl]
                        else:
                            ix = self._word_embedder.get_placeholder(ix)
                            placeholders[wl] = ix

                    question_words[question_ix, word_ix] = ix
                if self._char_emb is not None:
                    for char_ix, char in enumerate(word):
                        if char_ix == self.max_char_dim:
                            break
                        question_chars[question_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

        feed_dict[self.answer_paragraph] = batch.answer
        return feed_dict


class ParagraphQuestionGroupFeaturizedEncoder(Configurable):
    def __init__(self, featurizers: List[QuestionParagraphFeaturizer]):
        self.featurizers = featurizers
        self._max_char_dim = None

        self._group_feature_cache = {}
        self._word_embedder = None
        self._char_emb = None


        self.question_features = None
        self.question_words = None
        self.question_chars = None
        self.question_len = None
        self.context_len = None
        self.answer_paragraph = None

    def get_placeholders(self):
        return [x for x in [self.question_len, self.question_words,
                            self.question_chars, self.question_features,
                            self.context_len, self.answer_paragraph] if x is not None]

    def init(self, word_emb: WordEmbedder, char_emb: CharEmbedder):
        self._word_embedder = word_emb
        self._char_emb = char_emb

        if self._char_emb is not None:
            self._max_char_dim = self._char_emb.get_word_size_th()
        else:
            self._max_char_dim = 1

        self.question_words = tf.placeholder('int32', [None, None], name='question_words')
        self.question_len = tf.placeholder('int32', [None], name='question_len')
        self.context_len = tf.placeholder('int32', [None], name='context_len')

        self.question_features = tf.placeholder('float32',
                                                [None, None, None, sum(len(x.feature_names()) for x in self.featurizers)],
                                                name='question_features')

        if self._char_emb:
            self.question_chars = tf.placeholder('int32', [None, None, self._max_char_dim], name='question_chars')
        else:
            self.question_chars = None

        self.answer_paragraph = tf.placeholder('int32', [None], name='answer_start')

    def encode(self, batch: ParagraphQuestionGroup, is_train: bool):
        max_char_dim = self._max_char_dim
        n_questions = len(batch.questions)

        feed_dict = {}
        question_len = np.array([len(q) for q in batch.questions], dtype='int32')
        feed_dict[self.question_len] = question_len
        feed_dict[self.context_len] = np.array([sum(len(s) for s in para) for para in batch.paragraphs], dtype=np.int32)

        ques_word_dim = question_len.max()

        if self._word_embedder is not None:
            question_words = np.zeros([n_questions, ques_word_dim], dtype='int32')
            feed_dict[self.question_words] = question_words
        else:
            question_words = None

        if self._char_emb is not None:
            question_chars = np.zeros([n_questions, ques_word_dim, max_char_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
        else:
            question_chars = None

        for question_ix, question in enumerate(batch.questions):
            for word_ix, word in enumerate(question):
                if self._word_embedder is not None:
                    ix = self._word_embedder.question_word_to_ix(word)
                    if ix < 0:
                        raise ValueError("This encoder does not support placeholders")

                    question_words[question_ix, word_ix] = ix
                if self._char_emb is not None:
                    for char_ix, char in enumerate(word):
                        if char_ix == self._max_char_dim:
                            break
                        question_chars[question_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

        if batch.group_id in self._group_feature_cache:
            features = self._group_feature_cache[batch.group_id]
        else:
            # (featurizer, question, word, paragraph,, dim)
            features = [fe.build_features(batch.questions, batch.paragraphs) for fe in self.featurizers]

            features = transpose_lists(features)   # (question, featurizer, word, paragraph,, dim)
            features = [np.concatenate(x, axis=2) for x in features]  # (question, word, paragraph, dim)

            features = [np.pad(x, [(0, ques_word_dim-x.shape[0]), (0, 0), (0, 0)],  # pad word to `ques_word_dim`
                               mode="constant", constant_values=0) for x in features]
            features = np.stack(features, axis=0)  # stack to make it pure numpy

            if batch.group_id is not None:
                self._group_feature_cache[batch.group_id] = features
        feed_dict[self.question_features] = features

        feed_dict[self.answer_paragraph] = batch.answer
        return feed_dict

    def __getstate__(self):
        return dict(version=self.version, featurizers=self.featurizers)

    def __setstate__(self, state):
        self.__init__(state["featurizers"])
