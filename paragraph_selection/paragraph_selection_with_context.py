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
from encoder import QuestionEncoder, DocumentAndQuestionEncoder, MultiContextAndQuestionEncoder
from model import Model, ModelOutput, Prediction
from nn.embedder import WordEmbedder, CharWordEmbedder
from nn.layers import SequenceEncoder, SequenceMapper, get_keras_initialization, MergeLayer, AttentionMapper, Mapper, \
    SequenceMultiEncoder
from nn.ops import VERY_NEGATIVE_NUMBER
from paragraph_selection.paragraph_selection_featurizer import JointParagraphSelectionFeaturizer, \
    ParagraphSelectionFeaturizer
from paragraph_selection.paragraph_selection_model import FeaturizeredParagraph, FilteredData, \
    ParagraphSelectionPredictor
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import TriviaQaQuestion
from utils import ResourceLoader


class FeaturizedParagraphText(FeaturizeredParagraph):
    __slots__ = ["_context"]

    def __init__(self, para: FeaturizeredParagraph, context):
        super().__init__(para.question_id, para.doc_id, para.question, para.word_features, para.features, para.spans, para.answer)
        self._context = context

    @property
    def context(self):
        out = self._context
        self._context = None
        return out


class SelectionWithContextDataset(Dataset):
    def __init__(self, data: List[FeaturizeredParagraph], n_total, evidence: TriviaQaEvidenceCorpusTxt,
                 batching: ListBatcher, context_voc):
        self.data = data
        self.evidence = evidence
        self.n_total = n_total
        self.batching = batching
        self.context_voc = context_voc

    def get_vocab(self):
        voc = self.context_voc
        for point in self.data:
            voc.update(point.question)
        return voc

    @property
    def batch_size(self):
        return self.batching.get_fixed_batch_size()

    def percent_filtered(self):
        return (self.n_total - len(self.data)) / self.n_total

    def get_samples(self, n_samples: int):
        n_batches = len(self.data) // n_samples
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        for batch in self.batching.get_epoch(self.data):
            for i, para in enumerate(batch):
                read_to = para.spans.max()
                text = self.evidence.get_document(para.doc_id, read_to, flat=True)
                paragraphs = [text[s:e+1] for s,e in para.spans]
                batch[i] = FeaturizedParagraphText(para, paragraphs)
            yield batch

    def __len__(self):
        return self.batching.epoch_size(len(self.data))


class SelectionWithContextDatasetBuilder(DatasetBuilder):
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
        return SelectionWithContextDataset(data.data, true_len, evidence,
                                           self.train_batching if is_train else self.test_batching,
                                           data.voc)

    def build_stats(self, data: FilteredData) -> object:
        return None


class ParagraphSelectionModel(Model):
    """ Base class for models that use the question text, and manually built features for
     built for each (question, question_word, paragraph) and each (question, paragraph) set """

    def __init__(self,
                 encoder: MultiContextAndQuestionEncoder,
                 word_embed: Optional[WordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer):
        self.encoder = encoder
        self.featurizer = featurizer
        self.word_embed = word_embed

        self._q_mask = None
        self._n_paragraphs = None
        self._features = None
        self._word_features = None
        self._answer = None
        self._is_train = None

    def init(self, corpus, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader, [])

    def encode(self, examples: List[FeaturizedParagraphText], is_train: bool):
        para_dim = max(x.n_paragraphs for x in examples)
        word_dim = max(len(x.question) for x in examples)

        feed_dict = self.encoder.encode(examples, is_train)

        answer = np.zeros((len(examples), para_dim), dtype=np.bool)
        features = np.zeros((len(examples), para_dim, self.featurizer.n_features), dtype=np.float32)
        word_features = np.zeros((len(examples), para_dim, word_dim, self.featurizer.n_word_features), dtype=np.float32)
        n_paragraphs = np.zeros((len(examples)), dtype=np.int32)
        for i, ex in enumerate(examples):
            p = ex.n_paragraphs
            n_paragraphs[i] = p
            word_features[i, :p, :len(ex.question)] = ex.word_features
            features[i, :p] = ex.features
            answer[i, :p] = ex.answer

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
        if self.encoder is not None:
            self.encoder.init(batch_size, self.word_embed)
            self._q_mask = self.encoder.question_len
        else:
            self._q_mask = tf.placeholder('int32', [batch_size], name='question_len')

        self._is_train = tf.placeholder(tf.bool, ())
        self._answer = tf.placeholder(tf.bool, (batch_size, None))
        self._n_paragraphs = tf.placeholder(tf.int32, [batch_size])
        self._features = tf.placeholder(tf.float32, [batch_size, None, self.featurizer.n_features])
        self._word_features = tf.placeholder(tf.float32, [batch_size, None, None, self.featurizer.n_word_features])

    def get_placeholders(self):
        if self.encoder is None:
            placeholders = [self._q_mask]
        else:
            placeholders = self.encoder.get_placeholders()
        placeholders += [self._n_paragraphs, self._features, self._word_features, self._answer, self._is_train]
        return placeholders

    def get_predictions_for(self, input_tensors) -> ModelOutput:
        is_train = input_tensors[self._is_train]
        enc = self.encoder
        q_mask = input_tensors[enc.question_len]
        c_mask = input_tensors[enc.context_len]

        with tf.variable_scope("word-embed"):
            q_embed, c_embed = self.word_embed.embed(is_train,
                                        (input_tensors[enc.question_words], q_mask),
                                        (input_tensors[enc.context_words], c_mask))

        return self._get_prediction(is_train, q_embed, q_mask,
                                    c_embed, c_mask,
                                    input_tensors[self._features],
                                    input_tensors[self._word_features],
                                    input_tensors[self._n_paragraphs],
                                    input_tensors[self._answer])

    def _get_prediction(self, is_train, q_embed, q_mask, c_embed, c_mask, features, word_features, n_paragraphs, answer):
        raise NotImplementedError()


class DocumentEncoder(ParagraphSelectionModel):
    """ Merge the question embedding with the question features in a more generic way """

    def __init__(self,
                 word_embed: Optional[WordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer,
                 map_question: SequenceMapper,
                 map_context: SequenceMapper,
                 encode_question_words: Optional[SequenceMapper],
                 encode_context: Union[SequenceMultiEncoder, SequenceEncoder],
                 question_features: SequenceMapper,
                 merge_with_features: MergeLayer,
                 encode_joint_features: SequenceEncoder,
                 map_joint: SequenceMapper,
                 process: SequenceMapper,
                 predictor: ParagraphSelectionPredictor,
                 any_features: bool):
        super().__init__(MultiContextAndQuestionEncoder(), word_embed, featurizer)
        self.any_features = any_features
        self.encode_question_words = encode_question_words
        self.map_question = map_question
        self.map_context = map_context
        self.map_joint = map_joint
        self.encode_context = encode_context
        self.question_features = question_features
        self.process = process
        self.predictor = predictor
        self.merge_with_features = merge_with_features
        self.encode_joint_features = encode_joint_features

    def _get_prediction(self, is_train, q_embed, q_mask,  c_embed, c_mask, features, word_features, n_paragraphs, answer):
        s = tf.shape(word_features)
        batch = s[0]
        para_dim = s[1]
        word_dim = s[2]

        if self.any_features:
            word_features = tf.concat([word_features, tf.cast(word_features > 0, tf.float32)], axis=3)

        if self.map_question is not None:
            with tf.variable_scope("map_question"):
                q_embed = self.map_question.apply(is_train, q_embed, q_mask)

        flat_context = tf.reshape(c_embed, (-1, tf.shape(c_embed)[2], c_embed.shape.as_list()[-1]))
        flat_context_mask = tf.reshape(c_mask, (-1,))

        if self.map_context is not None:
            with tf.variable_scope("map_context"):
                flat_context = self.map_context.apply(is_train, flat_context, flat_context_mask)

        with tf.variable_scope("encode_context"):
            encoded_context = self.encode_context.apply(is_train, flat_context, flat_context_mask)
            n_encodes = 1 if len(encoded_context.shape) == 2 else encoded_context.shape.as_list()[1]
            encoded_context = tf.reshape(encoded_context, (batch, para_dim, n_encodes, encoded_context.shape.as_list()[-1]))

        if self.encode_question_words is not None:
            with tf.variable_scope("encode_questions"):
                q_encoded = self.encode_question_words.apply(is_train, q_embed, q_mask)
        else:
            q_encoded = q_embed

        context_matching_features = tf.einsum("qwf,qpdf->qpwd", q_encoded, encoded_context)
        word_features = tf.concat([word_features, context_matching_features], axis=3)

        if self.question_features is not None:
            with tf.variable_scope("question_features"):
                q_embed = self.question_features.apply(is_train, q_embed, q_mask)

        # (batch * paragraph, word, features) -> (batch*paragraph, features)
        word_features = tf.reshape(word_features, (-1, word_dim, word_features.shape.as_list()[-1]))
        q_embed = tf.tile(tf.expand_dims(q_embed, 1), [1, para_dim, 1, 1])
        q_embed = tf.reshape(q_embed, (-1, word_dim, q_embed.shape.as_list()[-1]))

        with tf.variable_scope("merge"):
            combined_fe = self.merge_with_features.apply(is_train, q_embed, word_features)

        flattened_mask = tf.reshape(tf.tile(tf.expand_dims(q_mask, 1), [1, para_dim]), (-1,))
        flattened_mask *= tf.cast(tf.reshape(tf.sequence_mask(n_paragraphs, para_dim), (-1,)), tf.int32)

        if self.map_joint is not None:
            with tf.variable_scope("map_joint"):
                combined_fe = self.map_joint.apply(is_train, combined_fe, flattened_mask)

        with tf.variable_scope("reduce_word_features"):
            combined_fe = self.encode_joint_features.apply(is_train, combined_fe, flattened_mask)

            # (batch*paragraph, features) -> (batch, paragraph, features)
            combined_fe = tf.reshape(combined_fe, (batch, para_dim, combined_fe.shape.as_list()[-1]))

        all_features = tf.concat([combined_fe, features], axis=2)

        with tf.variable_scope("process_features"):
            all_features = self.process.apply(is_train, all_features)

        with tf.variable_scope("predict"):
            return self.predictor.apply(all_features, answer, n_paragraphs)


class ContextTriAttention(ParagraphSelectionModel):
    """ Merge the question embedding with the question features in a more generic way """

    def __init__(self,
                 word_embed: Optional[WordEmbedder],
                 featurizer: ParagraphSelectionFeaturizer,
                 map_question: SequenceMapper,
                 map_atten: Optional[SequenceMapper],
                 merge_with_features: MergeLayer,
                 map_joint: SequenceMapper,
                 encode_word_features: SequenceEncoder,
                 process: SequenceMapper,
                 predictor: ParagraphSelectionPredictor,
                 any_features: bool):
        super().__init__(MultiContextAndQuestionEncoder(), word_embed, featurizer)
        self.any_features = any_features
        self.map_question = map_question
        self.map_atten = map_atten
        self.map_joint = map_joint
        self.encode_word_features = encode_word_features
        self.process = process
        self.predictor = predictor
        self.merge_with_features = merge_with_features

    def _get_prediction(self, is_train, q_embed, q_mask,  c_embed, c_mask, features, word_features, n_paragraphs, answer):
        if self.any_features:
            word_features = tf.concat([word_features, tf.cast(word_features > 0, tf.float32)], axis=3)

        # (batch, paragraph, word, features) -> (batch * paragraph, word, features)
        s = tf.shape(word_features)
        batch = s[0]
        para_dim = s[1]
        word_dim = s[2]

        with tf.variable_scope("map_atten"):
            q_atten = self.map_atten.apply(is_train, q_embed, q_mask)

        with tf.variable_scope("map_atten", reuse=True):
            if isinstance(self.map_atten, Mapper):
                c_atten = self.map_atten.apply(is_train, c_embed, c_mask)
            else:
                c_atten = self.map_atten.apply(is_train, tf.reshape(c_embed,
                                                                      (-1, tf.shape(c_embed)[2], c_embed.shape.as_list()[-1])),
                                                 tf.reshape(c_mask, (-1,)))
                c_atten = tf.reshape(c_atten, (batch, para_dim, -1, c_atten.shape.as_list()[-1]))

        project_dim = q_atten.shape.as_list()[-1]
        dot_w = tf.get_variable("dot_w", (1, 1, project_dim))
        dot_scores = tf.einsum("qif,qpjf->qpij", q_atten*dot_w, c_atten)

        q_w = tf.get_variable("q_w", project_dim)
        dot_scores += tf.expand_dims(tf.expand_dims(tf.tensordot(q_atten, q_w, [[2], [0]]), 1), 3)

        c_w = tf.get_variable("c_w", project_dim)
        dot_scores += tf.expand_dims(tf.tensordot(c_atten, c_w, [[3], [0]]), 2)

        dot_scores /= tf.cast(project_dim, tf.float32)

        c_flat_mask = tf.sequence_mask(tf.reshape(c_mask, (-1,)), tf.shape(c_embed)[2])
        c_flat_mask = tf.reshape(c_flat_mask, (batch, para_dim, -1))  # (batch, para, c_word)
        q_flat_mask = tf.tile(tf.expand_dims(q_mask, 1), [1, para_dim])
        q_flat_mask = tf.sequence_mask(tf.reshape(q_flat_mask, (-1,)), word_dim)
        q_flat_mask = tf.reshape(q_flat_mask, (batch, para_dim, -1))  # (batch, para, q_word)

        dot_mask = tf.logical_and(tf.expand_dims(c_flat_mask, 2), tf.expand_dims(q_flat_mask, 3))  # (batch, para, q_word, c_word)

        dot_scores = tf.exp(dot_scores + VERY_NEGATIVE_NUMBER * tf.cast(tf.logical_not(dot_mask), tf.float32))

        c_features = tf.einsum("qpij,qpjf->qpif", dot_scores, c_embed)
        word_features = tf.concat([word_features, c_features], axis=3)

        if self.map_question is not None:
            with tf.variable_scope("map_question"):
                q_embed = self.map_question.apply(is_train, q_embed, q_mask)

        q_embed = tf.tile(tf.expand_dims(q_embed, 1), [1, para_dim, 1, 1])
        word_features = tf.reshape(word_features, (-1, word_dim, word_features.shape.as_list()[-1]))
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
