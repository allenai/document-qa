from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import sys

from configurable import Configurable
from data_processing.document_splitter import DocumentSplitter, ParagraphFilter
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder, TextDataset, \
    LazyCorpusStatistics
from dataset import Dataset, ListBatcher
from encoder import QuestionEncoder
from model import Model, ModelOutput, Prediction
from nn.embedder import WordEmbedder, CharWordEmbedder
from nn.layers import SequenceEncoder, SequenceMapper, get_keras_initialization, MergeLayer
from nn.ops import VERY_NEGATIVE_NUMBER
from paragraph_selection.paragraph_selection_featurizer import JointParagraphSelectionFeaturizer, \
    ParagraphSelectionFeaturizer
from trivia_qa.read_data import TriviaQaQuestion
from utils import ResourceLoader, flatten_iterable


class FilteredData(object):
    def __init__(self, data: List, true_len: int, no_answer_pruned, no_answer_split_pruned, voc):
        self.data = data
        self.no_answer_split_pruned = no_answer_split_pruned
        self.no_answer_pruned = no_answer_pruned
        self.true_len = true_len
        self.voc = voc

    def __add__(self, other):
        return FilteredData(self.data + other.data, self.true_len + other.true_len,
                            self.no_answer_pruned + other.no_answer_pruned,
                            self.no_answer_split_pruned + other.no_answer_split_pruned,
                            self.voc.union(other.voc))


class FeaturizeredParagraph(object):
    __slots__ = ["question_id", "doc_id", "question", "word_features", "features", "spans", "answer"]

    def __init__(self, question_id: str, doc_id: str, question: List[str], word_features: np.ndarray,
                 features: np.ndarray, spans, answer: np.ndarray):
        self.question_id = question_id
        self.doc_id = doc_id
        self.question = question
        self.word_features = word_features
        self.features = features
        self.spans = spans
        self.answer = answer

    @property
    def n_paragraphs(self):
        return len(self.answer)

    def get_text(self):
        return self.question


class NParagraphsSortKey(Configurable):
    def __call__(self, p: FeaturizeredParagraph):
        return len(p.spans)


class ParagraphSelectionFeaturizer(Preprocessor):
    def __init__(self,
                 splitter: DocumentSplitter,
                 para_filter: Optional[ParagraphFilter],
                 word_featurizers: List[JointParagraphSelectionFeaturizer],
                 featurizers: List[ParagraphSelectionFeaturizer],
                 prune_no_answer: bool,
                 filter_initial_zeros: bool,
                 intern: bool=False,
                 context_voc: bool=False
                 ):
        self.intern = intern
        self.filter_initial_zeros = filter_initial_zeros
        self.splitter = splitter
        self.para_filter = para_filter
        self.featurizers = featurizers
        self.word_feautrizers = word_featurizers
        self.prune_no_answer = prune_no_answer
        self.context_voc = context_voc
        self.n_features = sum(len(x.get_feature_names()) for x in word_featurizers) + \
                          sum(len(x.get_feature_names()) for x in featurizers)
        self.n_word_features = sum(len(x.get_word_feature_names()) for x in word_featurizers)

    def finalize(self, data: FilteredData):
        if self.intern:
            question_map = {}
            for q in data.data:
                q.question_id = sys.intern(q.question_id)
                if q.question_id in question_map:
                    q.question = question_map[q.question_id]
                else:
                    q.question = tuple(sys.intern(w) for w in q.question)
                    question_map[q.question_id] = q.question

    def preprocess(self, questions: List[TriviaQaQuestion], evidence):
        out = []
        true_len = 0
        no_answer_pruned = 0
        no_answer_split_pruned = 0
        voc = set()
        for i, question in enumerate(questions):
            paragraphs = []
            doc_ids = []
            for doc in question.all_docs:
                true_len += 1
                if self.prune_no_answer and len(doc.answer_spans) == 0:
                    no_answer_pruned += 1
                    continue
                text = evidence.get_document(doc.doc_id)
                split = self.splitter.split_annotated(text, doc.answer_spans)
                if self.para_filter is not None:
                    split = self.para_filter.prune(question.question, split)
                if self.prune_no_answer and not any(len(x.answer_spans) > 0 for x in split):
                    no_answer_split_pruned += 1
                    continue
                doc_ids.append(doc.doc_id)
                paragraphs.append(split)

            if len(paragraphs) == 0:
                continue

            if self.context_voc:
                for doc in paragraphs:
                    for para in doc:
                        for sent in para.text:
                            voc.update(sent)

            paragraph_features = []
            word_features = []
            for ix, fe in enumerate(self.word_feautrizers):
                word, para = fe.get_joint_features(question.question, paragraphs)
                if para is not None:
                    paragraph_features.append(para)
                word_features.append(word)
            for fe in self.featurizers:
                paragraph_features.append(fe.get_features(question.question, paragraphs))

            if self.filter_initial_zeros and np.all(word_features[0] == 0):
                raise ValueError()

            if len(paragraph_features) > 0:
                paragraph_features = np.concatenate([np.array(x) for x in paragraph_features], axis=1)
            else:
                paragraph_features = np.zeros((len(word_features), 0), dtype=np.float32)

            if len(word_features) > 0:
                word_features = np.concatenate([np.array(x) for x in word_features], axis=2)
            else:
                word_features = np.zeros((len(paragraph_features), len(question.question), 0), dtype=np.float32)

            on_ix = 0
            for para, doc_id in zip(paragraphs, doc_ids):
                spans = np.array([(x.start, x.end) for x in para], dtype=np.int32)
                end = on_ix + len(para)
                out.append(FeaturizeredParagraph(
                    question.question_id,
                    doc_id,
                    question.question,
                    word_features[on_ix:end],
                    paragraph_features[on_ix:end],
                    spans,
                    np.array([len(x.answer_spans) for x in para], dtype=np.int32)
                ))
                on_ix = end
        return FilteredData(out, true_len, no_answer_pruned, no_answer_split_pruned, voc)

    def __setstate__(self, state):
        if "intern" not in state["state"]:
            state["state"]["intern"] = False
        if "context_voc" not in state["state"]:
            state["state"]["context_voc"] = False
        super().__setstate__(state)


class SelectionDatasetBuilder(DatasetBuilder):
    def __init__(self, train_batching: ListBatcher, test_batching: ListBatcher):
        self.train_batching = train_batching
        self.test_batching = test_batching

    def build_dataset(self, data: FilteredData, evidence, is_train: bool) -> Dataset:
        true_len = data.true_len
        n_examples = len(data.data)
        other_pruned = (true_len - n_examples) - data.no_answer_pruned - data.no_answer_split_pruned
        print("Building dataset with %d/%d (%.4f) examples," % (n_examples, true_len, n_examples/true_len))
        print("Pruned (%d (%.4f) non-answer, %d (%.4f) split, %d (%.4f) other)" % (
            data.no_answer_pruned, data.no_answer_pruned/true_len,
            data.no_answer_split_pruned, data.no_answer_split_pruned / true_len,
            other_pruned, other_pruned / true_len
        ))
        return TextDataset(data.data, true_len, self.train_batching if is_train else self.test_batching)

    def build_stats(self, data: FilteredData) -> object:
        return LazyCorpusStatistics(data.data)


class ParagraphPrediction(Prediction):
    def __init__(self, paragraph_scores):
        self.paragraph_scores = paragraph_scores
        self.top_paragraph_scores, self.top_paragraphs = tf.nn.top_k(paragraph_scores, tf.shape(paragraph_scores)[1], sorted=True)


class ParagraphSelectionPredictor(Configurable):
    """ Make a prediction from fixed length representations of a set of paragraphs """
    def apply(self, paragraph_features, answer, mask) -> ModelOutput:
        raise NotImplementedError()


class SigmoidPredictions(ParagraphSelectionPredictor):
    def __init__(self, init="glorot_uniform"):
        self.init = init

    def apply(self, paragraph_features, answer, mask):
        logits = fully_connected(paragraph_features, 1, activation_fn=None,
                                  weights_initializer=get_keras_initialization(self.init))
        logits = tf.squeeze(logits, axis=2)
        answer_mask = tf.cast(tf.sequence_mask(mask, tf.shape(paragraph_features)[1]), tf.float32)
        logits = VERY_NEGATIVE_NUMBER * (1 - answer_mask) + answer_mask * logits
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(answer, tf.float32), logits=logits)
        return ModelOutput(tf.reduce_mean(loss), ParagraphPrediction(logits))


class SoftmaxPrediction(ParagraphSelectionPredictor):
    def __init__(self, init="glorot_uniform", aggregate="sum"):
        self.init = init
        self.aggregate = aggregate

    def apply(self, paragraph_features, answer, mask):
        logits = fully_connected(paragraph_features, 1, activation_fn=None,
                                  weights_initializer=get_keras_initialization(self.init))
        logits = tf.squeeze(logits, axis=2)
        logit_mask = tf.cast(tf.sequence_mask(mask, tf.shape(paragraph_features)[1]), tf.float32)
        logits = VERY_NEGATIVE_NUMBER * (1 - logit_mask) + logit_mask * logits

        log_norms = tf.reduce_logsumexp(logits, axis=1)
        answer = tf.cast(answer, tf.float32)
        answer_logits = VERY_NEGATIVE_NUMBER * (1 - answer) + answer * logits

        if self.aggregate == "max":
            log_scores = tf.reduce_max(answer_logits, axis=1)
        elif self.aggregate == "sum":
            log_scores = tf.reduce_logsumexp(answer_logits, axis=1)
        else:
            raise ValueError()

        loss = -(log_scores - log_norms)
        return ModelOutput(tf.reduce_mean(loss), ParagraphPrediction(logits))


class ParagraphSelectionModel(Model):
    """ Base class for models that use the question text, and manually built features for
     built for each (question, question_word, paragraph) and each (question, paragraph) set """

    def __init__(self,
                 encoder: Optional[QuestionEncoder],
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer):
        self.encoder = encoder
        self.featurizer = featurizer
        self.word_embed = word_embed
        self.char_embed = char_embed

        self._q_mask = None
        self._n_paragraphs = None
        self._features = None
        self._word_features = None
        self._answer = None
        self._is_train = None

    def init(self, corpus, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader, [])
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(corpus)

    def encode(self, examples: List[FeaturizeredParagraph], is_train: bool):
        para_dim = max(x.n_paragraphs for x in examples)
        word_dim = max(len(x.question) for x in examples)

        if self.encoder is not None:
            feed_dict = self.encoder.encode([x.question for x in examples], is_train)
        else:
            q_mask = np.array([len(x.question) for x in examples], dtype=np.int32)
            feed_dict = {self._q_mask: q_mask}

        answer = np.zeros((len(examples), para_dim), dtype=np.bool)
        features = np.zeros((len(examples), para_dim, self.featurizer.n_features), dtype=np.float32)
        word_features = np.zeros((len(examples), para_dim, word_dim, self.featurizer.n_word_features), dtype=np.float32)
        n_paragraphs = np.zeros((len(examples)), dtype=np.int32)
        for i, ex in enumerate(examples):
            p = ex.n_paragraphs
            n_paragraphs[i] = p
            word_features[i, :p, :len(ex.question)] = ex.word_features
            features[i, :p] = ex.features
            answer[i, :p] = ex.answer > 0

        feed_dict[self._n_paragraphs] = n_paragraphs
        feed_dict[self._features] = features
        feed_dict[self._word_features] = word_features
        feed_dict[self._answer] = answer
        feed_dict[self._is_train] = is_train
        return feed_dict

    def set_inputs(self, datasets: List[Dataset], resource_loader: ResourceLoader):
        voc = set()
        for dataset in datasets:
            voc.update(dataset.get_vocab())

        batch_size = datasets[0].batch_size
        for dataset in datasets[1:]:
            if dataset.batch_size != batch_size:
                batch_size = None

        self.set_input_spec(batch_size, voc, resource_loader)

    def set_input_spec(self, batch_size, voc, resource_loader):
        if self.word_embed is not None:
            self.word_embed.init(resource_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(resource_loader, voc)
        if self.encoder is not None:
            self.encoder.init(batch_size, self.word_embed,
                              None if self.char_embed is None else self.char_embed.embeder)
            self._q_mask = self.encoder.question_len
        else:
            self._q_mask = tf.placeholder('int32', [batch_size], name='question_len')

        self._is_train = tf.placeholder(tf.bool, ())
        self._answer = tf.placeholder(tf.bool, (batch_size, None))
        self._n_paragraphs = tf.placeholder(tf.int32, [batch_size])
        self._features = tf.placeholder(tf.float32, [batch_size, None, self.featurizer.n_features])
        self._word_features = tf.placeholder(tf.float32, [batch_size, None, None, self.featurizer.n_word_features])

    def get_prediction(self) -> ModelOutput:
        if self.encoder is None:
            placeholders = [self._q_mask]
        else:
            placeholders = self.encoder.get_placeholders()
        placeholders += [self._n_paragraphs, self._features, self._word_features, self._answer]
        return self.get_predictions_for(self._is_train, {x:x for x in placeholders})

    def get_predictions_for(self, is_train, input_tensors) -> ModelOutput:
        enc = self.encoder
        q_mask = input_tensors[self._q_mask]

        if enc is not None:
            q_embed = []
            if enc.question_chars in input_tensors:
                with tf.variable_scope("char-embed"):
                    q = self.char_embed.embed(is_train, (input_tensors[enc.question_chars], q_mask))[0]
                q_embed.append(q)

            if enc.question_words in input_tensors:
                with tf.variable_scope("word-embed"):
                    q = self.word_embed.embed(is_train, (input_tensors[enc.question_words], q_mask))[0]
                q_embed.append(q)

            q_embed = tf.concat(q_embed, axis=2)
        else:
            q_embed = None

        return self._get_prediction(is_train, q_embed, q_mask,
                                    input_tensors[self._features],
                                    input_tensors[self._word_features],
                                    input_tensors[self._n_paragraphs],
                                    input_tensors[self._answer])

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_features, n_paragraphs, answer):
        raise NotImplementedError()


class FeaturersOnly(ParagraphSelectionModel):
    """ Ignore the question words """
    def __init__(self,
                 featurizer: ParagraphSelectionFeaturizer,
                 encode_word_features: SequenceEncoder,
                 process: SequenceMapper,
                 predictor: ParagraphSelectionPredictor,
                 any_features: bool=False
                 ):
        super().__init__(None, None, None, featurizer)
        self.encode_word_features = encode_word_features
        self.process = process
        self.predictor = predictor
        self.any_features = any_features

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_features, n_paragraphs, answer):
        if self.any_features:
            word_features = tf.concat([word_features, tf.cast(word_features > 0, tf.float32)], axis=3)
        # (batch, paragraph, word, features) -> (batch * paragraph, word, features)
        s = tf.shape(word_features)
        batch = s[0]
        para_dim = s[1]
        word_dim = s[2]
        n_word_fe = word_features.shape.as_list()[-1]

        if n_word_fe > 0:

            # (batch * paragraph, word, features) -> (batch*paragraph, features)
            word_features = tf.reshape(word_features, (-1, word_dim, n_word_fe))

            with tf.variable_scope("reduce_word_features"):
                # TODO we could consider a segmented representation w/tf.bool_mask
                # tile the question mask to get [q_len1, q_len1, q_len1, ... q_len_n, q_len_n]
                flattened_mask = tf.reshape(tf.tile(tf.expand_dims(q_mask, 1), [1, para_dim]), (-1,))
                # now set the lengths for the (question, paragraph) pairs that do not exists to zero
                flattened_mask *= tf.cast(tf.reshape(tf.sequence_mask(n_paragraphs, para_dim), (-1,)), tf.int32)
                word_features = self.encode_word_features.apply(is_train, word_features, flattened_mask)

            # (batch*paragraph, features) -> (batch, paragraph, features)
            word_features = tf.reshape(word_features, (batch, para_dim, n_word_fe))
            all_features = tf.concat([word_features, features], axis=2)
        else:
            all_features = features

        with tf.variable_scope("process_features"):
            all_features = self.process.apply(is_train, all_features)

        with tf.variable_scope("predict"):
            return self.predictor.apply(all_features, answer, n_paragraphs)


class WeightedFeatures(ParagraphSelectionModel):
    """ Use the question to weight the word features, and optionally build a fixed-size
     representation of the question to mixin with the rest of the features """

    def __init__(self,
                 word_embed: WordEmbedder,
                 char_embed: Optional[CharWordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer,
                 map_question: Optional[SequenceMapper],
                 compute_weights: SequenceMapper,
                 encode_word_features: SequenceEncoder,
                 encode_question: Optional[SequenceEncoder],
                 process: SequenceMapper,
                 predictor: ParagraphSelectionPredictor,
                 project: bool=False
                 ):
        super().__init__(QuestionEncoder(), word_embed, char_embed, featurizer)
        self.map_question = map_question
        self.encode_question = encode_question
        self.compute_weights = compute_weights
        self.encode_word_features = encode_word_features
        self.process = process
        self.predictor = predictor
        self.project = project

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_features, n_paragraphs, answer):
        # (batch, paragraph, word, features) -> (batch * paragraph, word, features)
        s = tf.shape(word_features)
        batch = s[0]
        para_dim = s[1]
        word_dim = s[2]
        n_word_fe = word_features.shape.as_list()[-1]

        if self.map_question is not None:
            with tf.variable_scope("map_question"):
                q_embed = self.map_question.apply(is_train, q_embed, q_mask)

        with tf.variable_scope("compute_weights"):
            # (batch, word, features)
            weights = self.compute_weights.apply(is_train, q_embed, q_mask)

        if self.project:
            feature_project = tf.get_variable("project_word_features",
                                              (word_features.shape.as_list()[-1], self.project))
            word_features = tf.tensordot(word_features, feature_project, [[3], [0]])
            n_word_fe = self.project
        else:
            word_features *= tf.expand_dims(weights, 1)

        # (batch * paragraph, word, features) -> (batch*paragraph, features)
        word_features = tf.reshape(word_features, (-1, word_dim, n_word_fe))

        with tf.variable_scope("reduce_word_features"):
            # tile the question mask to get [q_len1, q_len1, q_len1, ... q_len_n, q_len_n]
            flattened_mask = tf.reshape(tf.tile(tf.expand_dims(q_mask, 1), [1, para_dim]), (-1,))
            # now set the lengths for the (question, paragraph) pairs that do not exists to zero
            flattened_mask *= tf.cast(tf.reshape(tf.sequence_mask(n_paragraphs, para_dim), (-1,)), tf.int32)
            word_features = self.encode_word_features.apply(is_train, word_features, flattened_mask)

            # (batch*paragraph, features) -> (batch, paragraph, features)
            word_features = tf.reshape(word_features, (batch, para_dim, n_word_fe))

        all_features = [word_features, features]

        if self.encode_question is not None:
            with tf.variable_scope("encode_question"):
                question_enc = self.encode_question.apply(is_train, q_embed, q_mask)
                question_enc = tf.tile(tf.expand_dims(question_enc, 1), [1, para_dim, 1])
                all_features.append(question_enc)

        all_features = tf.concat(all_features, axis=2)

        with tf.variable_scope("process_features"):
            all_features = self.process.apply(is_train, all_features)

        with tf.variable_scope("predict"):
            return self.predictor.apply(all_features, answer, n_paragraphs)


class EncodedFeatures(ParagraphSelectionModel):
    """ Merge the question embedding with the question features in a more generic way """

    def __init__(self,
                 word_embed: WordEmbedder,
                 char_embed: Optional[CharWordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer,
                 map_question: Optional[SequenceMapper],
                 merge_with_features: MergeLayer,
                 map_joint: Optional[SequenceMapper],
                 encode_word_features: SequenceEncoder,
                 process: SequenceMapper,
                 predictor: ParagraphSelectionPredictor,
                 any_features: bool
                 ):
        super().__init__(QuestionEncoder(), word_embed, char_embed, featurizer)
        self.any_features = any_features
        self.map_question = map_question
        self.merge_with_features = merge_with_features
        self.map_joint = map_joint
        self.encode_word_features = encode_word_features
        self.process = process
        self.predictor = predictor

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_features, n_paragraphs, answer):
        if self.any_features:
            # word_features = tf.concat([tf.log(word_features+1), tf.cast(word_features > 0, tf.float32)], axis=3)
            word_features = tf.concat([word_features, tf.cast(word_features > 0, tf.float32)], axis=3)
        # (batch, paragraph, word, features) -> (batch * paragraph, word, features)
        s = tf.shape(word_features)
        batch = s[0]
        para_dim = s[1]
        word_dim = s[2]
        n_word_fe = word_features.shape.as_list()[-1]

        if self.map_question is not None:
            with tf.variable_scope("map_question"):
                q_embed = self.map_question.apply(is_train, q_embed, q_mask)

        q_embed = tf.tile(tf.expand_dims(q_embed, 1), [1, para_dim, 1, 1])

        # (batch * paragraph, word, features) -> (batch*paragraph, features)
        word_features = tf.reshape(word_features, (-1, word_dim, n_word_fe))
        q_embed = tf.reshape(q_embed, (-1, word_dim, q_embed.shape.as_list()[-1]))

        with tf.variable_scope("merge"):
            combined_fe = self.merge_with_features.apply(is_train, q_embed, word_features)

        flattened_mask = tf.reshape(tf.tile(tf.expand_dims(q_mask, 1), [1, para_dim]), (-1,))
        flattened_mask *= tf.cast(tf.reshape(tf.sequence_mask(n_paragraphs, para_dim), (-1,)), tf.int32)

        if self.map_joint is not None:
            with tf.variable_scope("map_joint"):
                combined_fe = self.map_joint.apply(is_train, combined_fe, flattened_mask)

        with tf.variable_scope("reduce_word_features"):
            combined_fe = self.encode_word_features.apply(is_train, combined_fe, flattened_mask)

            # (batch*paragraph, features) -> (batch, paragraph, features)
            combined_fe = tf.reshape(combined_fe, (batch, para_dim, combined_fe.shape.as_list()[-1]))

        all_features = tf.concat([combined_fe, features], axis=2)

        with tf.variable_scope("process_features"):
            all_features = self.process.apply(is_train, all_features)

        with tf.variable_scope("predict"):
            return self.predictor.apply(all_features, answer, n_paragraphs)

