from typing import List, Optional, Dict

import numpy as np
import tensorflow as tf

from docqa.configurable import Configurable
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, ContextAndQuestion
from docqa.data_processing.span_data import ParagraphSpans, TokenSpans
from docqa.data_processing.text_features import QaTextFeautrizer
from docqa.nn.embedder import WordEmbedder, CharEmbedder
from docqa.nn.span_prediction_ops import to_packed_coordinates_np

"""
Classes to map python objects we want to classify into numpy arrays we can feed into Tensorflow,
e.i. to map (quesiton-context-answer) -> {tf.placeholder / numpy arrays}
"""


class AnswerEncoder(Configurable):
    """ Encode just the answer span """

    def init(self, batch_size, context_word_dim) -> None:
        raise NotImplementedError()

    def encode(self, batch_size, context_len, context_word_dim, batch) -> Dict:
        raise NotImplementedError()

    def get_placeholders(self) -> List:
        raise NotImplementedError()


class SingleSpanAnswerEncoder(AnswerEncoder):
    """ Encode the answer as integer coordinates, pick a random answer span if multiple spans exists """

    def __init__(self):
        self.answer_spans = None

    def get_placeholders(self) -> List:
        return [self.answer_spans]

    def init(self, batch_size, context_word_dim):
        self.answer_spans = tf.placeholder('int32', [batch_size, 2], name='answer_spans')

    def encode(self, batch_size, context_len, context_word_dim, batch) -> Dict:
        answer_spans = np.zeros([batch_size, 2], dtype='int32')

        for doc_ix, doc in enumerate(batch):
            answer = doc.answer

            if answer is None:
                continue

            if isinstance(answer, ParagraphSpans):
                answer = doc.answer[np.random.randint(0, len(doc.answer))]
                word_start = answer.para_word_start
                word_end = answer.para_word_end
            elif isinstance(answer, TokenSpans):
                candidates = np.where(answer.answer_spans[:, 1] < context_len[doc_ix])[0]
                if len(candidates) == 0:
                    continue
                ix = candidates[np.random.randint(0, len(candidates))]
                word_start, word_end = answer.answer_spans[ix]
            else:
                raise NotImplementedError()

            if word_start > word_end:
                raise ValueError()
            if word_end >= context_len[doc_ix]:
                raise ValueError(word_end)

            answer_spans[doc_ix, 0] = word_start
            answer_spans[doc_ix, 1] = word_end
        return {self.answer_spans: answer_spans}

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return self.__init__()


class DenseMultiSpanAnswerEncoder(AnswerEncoder):
    """ Encode the answer spans into bool (span_start) and (span_end) arrays """

    def __init__(self):
        self.answer_starts = None
        self.answer_ends = None

    def get_placeholders(self) -> List:
        return [self.answer_starts, self.answer_ends]

    def init(self, batch_size, context_word_dim):
        self.answer_starts = tf.placeholder('bool', [batch_size, context_word_dim], name='answer_starts')
        self.answer_ends = tf.placeholder('bool', [batch_size, context_word_dim], name='answer_ends')

    def encode(self, batch_size, context_len, context_word_dim, batch) -> Dict:
        answer_starts = np.zeros((batch_size, context_word_dim), dtype=np.bool)
        answer_ends = np.zeros((batch_size, context_word_dim), dtype=np.bool)
        for doc_ix, doc in enumerate(batch):
            if doc.answer is None:
                continue
            answer_spans = doc.answer.answer_spans
            answer_spans = answer_spans[answer_spans[:, 1] < context_word_dim]
            answer_starts[doc_ix, answer_spans[:, 0]] = True
            answer_ends[doc_ix, answer_spans[:, 1]] = True
        return {self.answer_starts: answer_starts, self.answer_ends: answer_ends}

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return DenseMultiSpanAnswerEncoder()


class GroupedSpanAnswerEncoder(AnswerEncoder):
    """ Encode the answer spans into bool (span_start) and (span_end) arrays, and also record
    the group_id if one is present in the answer. Used for the "shared-norm" approach """

    def __init__(self):
        self.answer_starts = None
        self.answer_ends = None
        self.group_ids = None

    def get_placeholders(self) -> List:
        return [self.answer_starts, self.answer_ends, self.group_ids]

    def init(self, batch_size, context_word_dim):
        self.answer_starts = tf.placeholder('bool', [batch_size, context_word_dim], name='answer_starts')
        self.answer_ends = tf.placeholder('bool', [batch_size, context_word_dim], name='answer_ends')
        self.group_ids = tf.placeholder('int32', [batch_size], name='group_ids')

    def encode(self, batch_size, context_len, context_word_dim, batch) -> Dict:
        has_group = hasattr(batch[0].answer, "group_id")
        answer_starts = np.zeros((batch_size, context_word_dim), dtype=np.bool)
        answer_ends = np.zeros((batch_size, context_word_dim), dtype=np.bool)
        group_id = np.zeros(batch_size, dtype=np.int32)
        for doc_ix, doc in enumerate(batch):
            if has_group:
                group_id[doc_ix] = doc.answer.group_id
            if doc.answer is None:
                continue
            answer_spans = doc.answer.answer_spans
            answer_spans = answer_spans[answer_spans[:, 1] < context_word_dim]

            answer_starts[doc_ix, answer_spans[:, 0]] = True
            answer_ends[doc_ix, answer_spans[:, 1]] = True
        if not has_group:
            group_id = np.arange(0, len(batch))
        return {self.answer_starts: answer_starts, self.answer_ends: answer_ends,
                self.group_ids:group_id}

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return GroupedSpanAnswerEncoder()


class PackedMultiSpanAnswerEncoder(AnswerEncoder):
    """ Records the span in a bool array in the packed format of `to_packed_coordinates` """

    def __init__(self, bound):
        self.bound = bound
        self.correct_spans = None

    def get_placeholders(self) -> List:
        return [self.correct_spans]

    def init(self, batch_size, context_word_dim):
        self.correct_spans = tf.placeholder('bool', [batch_size, None], name='correct_span')

    def encode(self, batch_size, context_len, context_word_dim, batch) -> Dict:
        sz = to_packed_coordinates_np(np.array([[context_word_dim-self.bound, context_word_dim-1]]),
                                   context_word_dim, self.bound)[0] + 1
        output = np.zeros((len(batch), sz), dtype=np.bool)
        for doc_ix, doc in enumerate(batch):
            output[doc_ix, to_packed_coordinates_np(doc.answer.answer_spans, context_word_dim, self.bound)] = True
        return {self.correct_spans: output}

    def __getstate__(self):
        return dict(bound=self.bound)

    def __setstate__(self, state):
        return PackedMultiSpanAnswerEncoder(state["bound"])


class DocumentAndQuestionEncoder(Configurable):
    """
    Uses a WordEmbedder/CharEmbedder (passed in by the client in `init`) to encode text into padded batches of arrays.
    It should really be called "ParagraphAndQuestionEncoder", but we are stuck with this for now
    """

    def __init__(self,
                 answer_encoder: AnswerEncoder,
                 doc_size_th: Optional[int]=None,
                 word_featurizer: Optional[QaTextFeautrizer]=None):
        # Parameters
        self.answer_encoder = answer_encoder
        self.doc_size_th = doc_size_th

        self.word_featurizer = word_featurizer

        self._word_embedder = None
        self._char_emb = None

        # Internal stuff we need to set on `init`
        self.len_opt = None
        self.batch_size = None
        self.max_context_word_dim = None
        self.max_ques_word_dim = None
        self.max_char_dim = None

        self.context_features = None
        self.context_words = None
        self.context_chars = None
        self.context_len = None
        self.context_word_len = None
        self.question_features = None
        self.question_words = None
        self.question_chars = None
        self.question_len = None
        self.question_word_len = None

    @property
    def version(self):
        return 3

    def init(self, input_spec: ParagraphAndQuestionSpec, len_op: bool,
             word_emb: WordEmbedder, char_emb: Optional[CharEmbedder]):

        self._word_embedder = word_emb
        self._char_emb = char_emb

        self.batch_size = input_spec.batch_size
        self.len_opt = len_op

        if self._char_emb is not None:
            if input_spec.max_word_size is not None:
                self.max_char_dim = min(self._char_emb.get_word_size_th(), input_spec.max_word_size)
            else:
                self.max_char_dim = self._char_emb.get_word_size_th()
        else:
            self.max_char_dim = 1

        if not self.len_opt:
            self.max_ques_word_dim = input_spec.max_num_quesiton_words
            self.max_context_word_dim = input_spec.max_num_context_words
            if self.max_ques_word_dim is None or self.max_context_word_dim is None:
                raise ValueError()
            if self.doc_size_th is not None:
                self.max_context_word_dim = min(self.max_context_word_dim, self.doc_size_th)
        else:
            self.max_ques_word_dim = None
            self.max_context_word_dim = None

        n_question_words = self.max_ques_word_dim
        n_context_words = self.max_context_word_dim
        batch_size = self.batch_size

        self.context_words = tf.placeholder('int32', [batch_size, n_context_words], name='context_words')
        self.context_len = tf.placeholder('int32', [batch_size], name='context_len')

        self.question_words = tf.placeholder('int32', [batch_size, n_question_words], name='question_words')
        self.question_len = tf.placeholder('int32', [batch_size], name='question_len')

        if self._char_emb:
            self.context_chars = tf.placeholder('int32', [batch_size, n_context_words, self.max_char_dim], name='context_chars')
            self.question_chars = tf.placeholder('int32', [batch_size, n_question_words, self.max_char_dim], name='question_chars')
            self.question_word_len = tf.placeholder('int32', [batch_size, n_question_words], name='question_len')
            self.context_word_len = tf.placeholder('int32', [batch_size, n_context_words], name='context_len')
        else:
            self.context_chars = None
            self.question_chars = None
            self.context_word_len = None
            self.question_word_len = None

        if self.word_featurizer is not None:
            self.question_features = tf.placeholder('float32',
                                                    [batch_size, n_question_words,
                                                     self.word_featurizer.n_question_features()],
                                                    name='question_features')
            self.context_features = tf.placeholder('float32', [batch_size, n_context_words,
                                                               self.word_featurizer.n_context_features()],
                                                   name='context_features')
        else:
            self.question_features = None
            self.context_features = None

        self.answer_encoder.init(batch_size, n_context_words)

    def get_placeholders(self):
        return [x for x in
                [self.question_len, self.question_words, self.question_chars, self.question_features,
                 self.context_len, self.context_words, self.context_chars, self.context_features,
                 self.question_word_len, self.context_word_len]
                if x is not None] + self.answer_encoder.get_placeholders()

    def encode(self, batch: List[ContextAndQuestion], is_train: bool):
        batch_size = len(batch)
        if self.batch_size is not None:
            if self.batch_size < batch_size:
                raise ValueError("Batch sized we pre-specified as %d, "
                                 "but got a batch of %d" % (self.batch_size, batch_size))
            # We have a fixed batch size, so we will pad our inputs with zeros along the batch dimension
            batch_size = self.batch_size

        context_word_dim, ques_word_dim, max_char_dim = \
            self.max_context_word_dim, self.max_ques_word_dim, self.max_char_dim

        feed_dict = {}

        # compute the question/word lengths
        if context_word_dim is not None:
            context_len = np.array([min(doc.n_context_words, context_word_dim)  # Might truncate context
                                    for doc in batch], dtype='int32')
        else:
            context_len = np.array([doc.n_context_words for doc in batch], dtype='int32')
            context_word_dim = context_len.max()

        question_len = np.array([len(x.question) for x in batch], dtype='int32')
        if ques_word_dim is not None:
            if question_len.max() > ques_word_dim:
                raise ValueError("Have a question of len %d but max ques dim is %d" %
                                 (question_len.max(), ques_word_dim))
        else:
            ques_word_dim = question_len.max()

        feed_dict[self.context_len] = context_len
        feed_dict[self.question_len] = question_len

        # Setup word placeholders
        if self._word_embedder is not None:
            context_words = np.zeros([batch_size, context_word_dim], dtype='int32')
            question_words = np.zeros([batch_size, ques_word_dim], dtype='int32')
            feed_dict[self.context_words] = context_words
            feed_dict[self.question_words] = question_words
        else:
            question_words, context_words = None, None

        # Setup char placeholders
        if self._char_emb is not None:
            context_chars = np.zeros([batch_size, context_word_dim, max_char_dim], dtype='int32')
            question_chars = np.zeros([batch_size, ques_word_dim, max_char_dim], dtype='int32')
            context_word_len = np.zeros([batch_size, context_word_dim], dtype='int32')
            question_word_len = np.zeros([batch_size, ques_word_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
            feed_dict[self.context_chars] = context_chars
            feed_dict[self.question_word_len] = question_word_len
            feed_dict[self.context_word_len] = context_word_len
        else:
            context_chars, question_chars, question_word_len, context_word_len = None, None, None, None

        query_once = self._word_embedder.query_once()

        # Now fill in the place holders by iterating through the data
        for doc_ix, doc in enumerate(batch):
            doc_mapping = {}  # word->ix mapping if `query_once` is True

            for word_ix, word in enumerate(doc.question):
                if self._word_embedder is not None:
                    if query_once:
                        ix = doc_mapping.get(word)
                        if ix is None:
                            ix = self._word_embedder.context_word_to_ix(word, is_train)
                            doc_mapping[word] = ix
                    else:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                    question_words[doc_ix, word_ix] = ix
                if self._char_emb is not None:
                    question_word_len[doc_ix, word_ix] = min(self.max_char_dim, len(word))
                    for char_ix, char in enumerate(word):
                        if char_ix == self.max_char_dim:
                            break
                        question_chars[doc_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

            for word_ix, word in enumerate(doc.get_context()):
                if self._word_embedder is not None:
                    if query_once:
                        ix = doc_mapping.get(word)
                        if ix is None:
                            ix = self._word_embedder.context_word_to_ix(word, is_train)
                            doc_mapping[word] = ix
                    else:
                        ix = self._word_embedder.context_word_to_ix(word, is_train)
                    context_words[doc_ix, word_ix] = ix

                if self._char_emb is not None:
                    context_word_len[doc_ix, word_ix] = min(self.max_char_dim, len(word))
                    for char_ix, char in enumerate(word):
                        if char_ix == self.max_char_dim:
                            break
                        context_chars[doc_ix, word_ix, char_ix] = self._char_emb.char_to_ix(char)

        # Answer placeholders
        feed_dict.update(self.answer_encoder.encode(batch_size, context_len, context_word_dim, batch))

        # Features placeholders
        if self.word_featurizer is not None:
            question_word_features = np.zeros((batch_size, ques_word_dim, self.word_featurizer.n_question_features()))
            context_word_features = np.zeros((batch_size, context_word_dim, self.word_featurizer.n_context_features()))
            for doc_ix, doc in enumerate(batch):
                q_f, c_f = self.word_featurizer.get_features(doc.question, doc.get_context())
                question_word_features[doc_ix, :q_f.shape[0]] = q_f
                context_word_features[doc_ix, :c_f.shape[0]] = c_f
            feed_dict[self.context_features] = context_word_features
            feed_dict[self.question_features] = question_word_features

        return feed_dict

    def __getstate__(self):
        # The placeholders are considered transient, the model
        # will be re-initailized when re-loaded
        state = dict(
            answer_encoder=self.answer_encoder,
            doc_size_th=self.doc_size_th,
            word_featurizer=self.word_featurizer,
            version=self.version
        )
        return state

    def __setstate__(self, state):
        if state["version"] == 0:
            if "word_featurizer" in state["state"]:
                raise ValueError()
            state["state"]["word_featurizer"] = None
        if state["version"] <= 1:
            if "answer_encoder" in state["state"]:
                raise ValueError()
            state["state"]["answer_encoder"] = SingleSpanAnswerEncoder()
        elif state["version"] <= 2:
            super().__setstate__(state)
        else:
            del state["version"]
            return self.__init__(**state)


class CheatingEncoder(DocumentAndQuestionEncoder):
    """ Useful for debugging, encodes where the correct answer span is in the context """

    def __init__(self, answer_encoder: AnswerEncoder):
        super().__init__(answer_encoder)

    def encode(self, batch: List[ContextAndQuestion], is_train: bool):
        batch_size = len(batch)
        if self.batch_size is not None:
            if self.batch_size < batch_size:
                raise ValueError()
            batch_size = self.batch_size
        N = batch_size
        # else dynamically use the batch size of the examples

        context_word_dim, ques_word_dim, max_char_dim = \
            self.max_context_word_dim, self.max_ques_word_dim, self.max_char_dim

        feed_dict = {}

        if is_train and context_word_dim is not None:
            # Context might be truncated
            context_len = np.array([min(doc.n_context_words, context_word_dim)
                                    for doc in batch], dtype='int32')
        else:
            context_len = np.array([doc.n_context_words for doc in batch], dtype='int32')
            context_word_dim = context_len.max()

        question_len = np.array([len(x.question) for x in batch], dtype='int32')
        if question_len.max() > ques_word_dim:
            raise ValueError("Have a question of len %d but max ques dim is %d" %
                             (question_len.max(), ques_word_dim))
        feed_dict[self.context_len] = context_len
        feed_dict[self.question_len] = question_len

        if self.len_opt:
            ques_word_dim = min(ques_word_dim, question_len.max())
            context_word_dim = min(context_word_dim, context_len.max())

        if self._word_embedder is not None:
            context_words = np.zeros([N, context_word_dim], dtype='int32')
            question_words = np.zeros([N, ques_word_dim], dtype='int32')
            feed_dict[self.context_words] = context_words
            feed_dict[self.question_words] = question_words
        else:
            question_words, context_words = None, None

        if self._char_emb is not None:
            context_chars = np.zeros([N, context_word_dim, max_char_dim], dtype='int32')
            question_chars = np.zeros([N, ques_word_dim, max_char_dim], dtype='int32')
            feed_dict[self.question_chars] = question_chars
            feed_dict[self.context_chars] = context_chars

        # Build vector encoding of the answers for each question in the batch
        for doc_ix, doc in enumerate(batch):
            context_words[doc_ix] = 0
            if doc.answer is None:
                continue

            for s, e in doc.answer.answer_spans:
                context_words[doc_ix, s] = self._word_embedder.question_word_to_ix("what", False)
                context_words[doc_ix, e] = self._word_embedder.question_word_to_ix("the", False)
                for i in range(s+1, e):
                    context_words[doc_ix, i] = self._word_embedder.question_word_to_ix("a", False)
                break

        feed_dict.update(self.answer_encoder.encode(batch_size, context_len, context_word_dim, batch))

        return feed_dict


