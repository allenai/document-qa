from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.layers import fully_connected
import sys

from configurable import Configurable
from data_processing.document_splitter import DocumentSplitter, ParagraphFilter
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder, TextDataset, \
    LazyCorpusStatistics
from dataset import Dataset
from encoder import QuestionEncoder, DocumentAndQuestionEncoder, MultiContextAndQuestionEncoder
from model import Model, Prediction
from nn.embedder import WordEmbedder, CharWordEmbedder
from nn.layers import SequenceEncoder, SequenceMapper, get_keras_initialization, MergeLayer, AttentionMapper, Mapper
from nn.ops import VERY_NEGATIVE_NUMBER
from paragraph_selection.paragraph_selection_featurizer import JointParagraphSelectionFeaturizer, \
    ParagraphSelectionFeaturizer
from experimental.paragraph_selection.paragraph_selection_model import FeaturizeredParagraph, FilteredData, \
    ParagraphSelectionPredictor
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import TriviaQaQuestion
from utils import ResourceLoader, flatten_iterable


class QuestionOccurance(object):

    def __init__(self, question_id, doc_id, question, occurances, features, spans, answer):
        self.question = question
        self.question_id = question_id
        self.doc_id = doc_id
        self.features = features
        self.occurances = occurances
        self.spans = spans
        self.answer = answer

    def get_text(self):
        return self.question

    @property
    def n_paragraphs(self):
        return len(self.answer)


class QuestionOccuranceData(object):
    def __init__(self, data: List[QuestionOccurance], true_len: int, no_answer_pruned, no_answer_split_pruned):
        self.data = data
        self.no_answer_split_pruned = no_answer_split_pruned
        self.no_answer_pruned = no_answer_pruned
        self.true_len = true_len

    def __add__(self, other):
        return QuestionOccuranceData(self.data + other.data, self.true_len + other.true_len,
                            self.no_answer_pruned + other.no_answer_pruned,
                            self.no_answer_split_pruned + other.no_answer_split_pruned)


class OccuranceFeaturizer(Preprocessor):
    def __init__(self,
                 splitter: DocumentSplitter,
                 para_filter: Optional[ParagraphFilter],
                 featurizers: List[ParagraphSelectionFeaturizer],
                 stop_words,
                 normalizer,
                 prune_no_answer: bool,
                 intern: bool=False):
        self.featurizers = featurizers
        self.intern = intern
        self.splitter = splitter
        self.para_filter = para_filter
        self.stop_words = stop_words
        self.normalizer = normalizer
        self.prune_no_answer = prune_no_answer

    @property
    def n_features(self):
        return sum(len(x.get_feature_names()) for x in self.featurizers)

    @property
    def n_match_categories(self):
        return 3

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
        stop = self.stop_words.words
        true_len = 0
        no_answer_pruned = 0
        no_answer_split_pruned = 0
        for i, question in enumerate(questions):
            feature_map = {}
            for ix, q in enumerate(question.question):
                if q.lower() in stop:
                    continue
                feature_map[q] = np.array([ix, 0], dtype=np.int16)
                q = q.lower()
                if q not in feature_map:
                    feature_map[q] = np.array([ix, 1], dtype=np.int16)
                q = self.normalizer.normalize(q)
                if q not in feature_map:
                    feature_map[q] = np.array([ix, 2], dtype=np.int16)

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

                all_occurrences = []
                for para in split:
                    occurrences = []
                    dist = 0

                    for word in flatten_iterable(para.text):
                        fe = feature_map.get(word)
                        if fe is None:
                            fe = feature_map.get(word.lower())
                        if fe is None:
                            fe = feature_map.get(self.normalizer.normalize(word))
                        if fe is not None:
                            if dist != 0:
                                occurrences.append(np.array([-dist, 0], dtype=np.int16))
                                dist = 0
                            occurrences.append(fe)
                        else:
                            dist += 1

                    # text = flatten_iterable(para.text)
                    # on_ix = 0
                    # for z, (ix, _) in enumerate(occurrences):
                    #     if ix < 0:
                    #         on_ix += -ix
                    #     else:
                    #         if self.normalizer.normalize(question.question[ix]) != \
                    #                 self.normalizer.normalize(text[on_ix]):
                    #             raise ValueError(z)
                    #         on_ix += 1

                    if dist > 0:
                        occurrences.append(np.array([-dist, 0], dtype=np.int16))
                    if len(occurrences) == 0:
                        all_occurrences.append(np.zeros((0, 2), np.int16))
                    else:
                        all_occurrences.append(np.stack(occurrences, axis=0))

                if len(self.featurizers) == 0:
                    paragraph_features = np.zeros((len(split), 0), dtype=np.float32)
                else:
                    paragraph_features = [f.get_features(question.question, split) for f in self.featurizers]
                    paragraph_features = np.concatenate(paragraph_features, axis=1)

                out.append(QuestionOccurance(question.question_id, doc.doc_id, question.question,
                                             all_occurrences, paragraph_features,
                                             np.array([(x.start, x.end) for x in split], np.int32),
                                             np.array([len(x.answer_spans) for x in split], np.int32)))

        return QuestionOccuranceData(out, true_len, no_answer_pruned, no_answer_split_pruned)


class OccuranceDatasetBuilder(DatasetBuilder):
    def __init__(self, train_batching: Batcher, test_batching: Batcher):
        self.train_batching = train_batching
        self.test_batching = test_batching

    def build_dataset(self, data: QuestionOccuranceData, evidence, is_train: bool) -> Dataset:
        true_len = data.true_len
        n_examples = len(data.data)
        other_pruned = (true_len - n_examples) - data.no_answer_pruned - data.no_answer_split_pruned
        print("Building dataset with %d/%d (%.4f) examples," % (n_examples, true_len, n_examples/true_len))
        print("Pruned (%d (%.4f) non-answer, %d (%.4f) split, %d (%.4f) other)" % (
            data.no_answer_pruned, data.no_answer_pruned/true_len,
            data.no_answer_split_pruned, data.no_answer_split_pruned / true_len,
            other_pruned, other_pruned / true_len
        ))
        return TextDataset(data.data, true_len-data.no_answer_pruned, self.train_batching if is_train else self.test_batching)

    def build_stats(self, data: FilteredData) -> object:
        return LazyCorpusStatistics(data.data)


class WordOccuranceSelectionModel(Model):
    """ Base class for models that use the question text, and manually built features for
     built for each (question, question_word, paragraph) and each (question, paragraph) set """

    def __init__(self,
                 encoder: Optional[QuestionEncoder],
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 featurizer: OccuranceFeaturizer):
        self.encoder = encoder
        self.featurizer = featurizer
        self.word_embed = word_embed
        self.char_embed = char_embed

        self._q_mask = None
        self._n_paragraphs = None
        self._features = None
        self._word_occurances = None
        self._occurance_lens = None
        self._answer = None
        self._is_train = None

    def init(self, corpus, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader, [])
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(corpus)

    def encode(self, examples: List[QuestionOccurance], is_train: bool):
        para_dim = max(x.n_paragraphs for x in examples)
        occurance_dim = max(max(len(o) for o in x.occurances) for x in examples)

        if self.encoder is not None:
            feed_dict = self.encoder.encode([x.question for x in examples], is_train)
        else:
            q_mask = np.array([len(x.question) for x in examples], dtype=np.int32)
            feed_dict = {self._q_mask: q_mask}

        answer = np.zeros((len(examples), para_dim), dtype=np.bool)
        features = np.zeros((len(examples), para_dim, self.featurizer.n_features), dtype=np.float32)
        occurances = np.zeros( (len(examples), para_dim, occurance_dim, 2), dtype=np.int16)
        occurance_len = np.zeros((len(examples), para_dim), dtype=np.int32)
        n_paragraphs = np.zeros((len(examples)), dtype=np.int32)
        for i, ex in enumerate(examples):
            p = ex.n_paragraphs
            occurance_len[i, :len(ex.occurances)] = [len(o) for o in ex.occurances]
            for para_ix, occ in enumerate(ex.occurances):
                occurances[i, para_ix, :len(occ)] = occ
            n_paragraphs[i] = p
            features[i, :p] = ex.features
            answer[i, :p] = ex.answer

        feed_dict[self._n_paragraphs] = n_paragraphs
        feed_dict[self._features] = features
        feed_dict[self._occurance_lens] = occurance_len
        feed_dict[self._word_occurances] = occurances
        feed_dict[self._answer] = answer
        feed_dict[self._is_train] = is_train
        return feed_dict

    def set_inputs(self, datasets: List[TextDataset], resource_loader: ResourceLoader):
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
        self._word_occurances = tf.placeholder(tf.int32, [batch_size, None, None, 2])
        self._occurance_lens = tf.placeholder(tf.int32, [batch_size, None])

    def get_prediction(self) -> ModelOutput:
        if self.encoder is None:
            placeholders = [self._q_mask]
        else:
            placeholders = self.encoder.get_placeholders()
        print(placeholders)
        placeholders += [self._n_paragraphs, self._features, self._word_occurances, self._occurance_lens, self._answer]
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
                                    input_tensors[self._word_occurances],
                                    input_tensors[self._occurance_lens],
                                    input_tensors[self._n_paragraphs],
                                    input_tensors[self._answer])

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_occurances, occurance_len, n_paragraphs, answer):
        raise NotImplementedError()


class EncodedOccurancePredictor(WordOccuranceSelectionModel):
    def __init__(self,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 featurizer: OccuranceFeaturizer,
                 question_map: Optional[SequenceMapper],
                 occurance_encoder: SequenceEncoder,
                 paragraph_encoder: SequenceMapper,
                 prediction_layer: ParagraphSelectionPredictor,
                 feature_vec_size: int,
                 distance_vecs: int):
        super().__init__(QuestionEncoder(), word_embed, char_embed, featurizer)
        self.question_map = question_map
        self.feature_vec_size = feature_vec_size
        self.occurance_encoder = occurance_encoder
        self.paragraph_mapper = paragraph_encoder
        self.distance_vecs = distance_vecs
        self.prediction_layer = prediction_layer

    def _get_prediction(self, is_train, q_embed, q_mask, features, word_occurances,
                        occurance_len, n_paragraphs, answer):
        s = tf.shape(word_occurances)
        batch = s[0]
        word_dim = tf.shape(q_embed)[1]
        occurance_dim = tf.shape(word_occurances)[2]

        if self.question_map is not None:
            with tf.variable_scope("map_question"):
                q_embed = self.question_map.apply(is_train, q_embed, q_mask)

        vec_dim = q_embed.shape.as_list()[-1]

        dist_vecs = tf.get_variable("dist_vecs", (self.distance_vecs, vec_dim))

        # Build the set of vectors we will need to index into, includes question-specific
        # word vector encodings and the distance vector encodings
        vectors = tf.reshape(q_embed, (-1, q_embed.shape.as_list()[-1]))
        vectors = tf.boolean_mask(vectors, tf.reshape(tf.sequence_mask(q_mask, word_dim), ((-1,))))
        vectors = tf.concat([tf.zeros((1, vec_dim)), vectors, dist_vecs], axis=0)

        word_ix = word_occurances[:, :, :, 0]
        # plus one offsets the zero vector, plus cumsum offsets to the correct word-specific vector
        offset_word_ix = 1 + word_ix + tf.expand_dims(tf.expand_dims(tf.cumsum(q_mask, exclusive=True), 1), 2)
        # Use either the question-specific offset, or the distance vec offset
        word_ix = tf.where(word_ix >= 0, offset_word_ix, 1 + tf.reduce_sum(q_mask) + tf.minimum(-word_ix-1, self.distance_vecs-1))

        # now a (question, paragraph, occurance_len, dim) sized matrix
        vectorizd_occurances = tf.gather(vectors, word_ix)

        feature_vecs = tf.get_variable("feature_vecs", (self.featurizer.n_match_categories, self.feature_vec_size))
        feature_vecs = tf.concat([tf.zeros((1, self.feature_vec_size)), feature_vecs], axis=0)
        vectorized_cat = tf.gather(feature_vecs, word_occurances[:, :, :, 1] + 1)
        vectorizd_occurances = tf.concat([vectorizd_occurances, vectorized_cat], axis=3)

        # now a (question * paragraph, occurance_len, dim) sized matrix
        vectorizd_occurances = tf.reshape(vectorizd_occurances, (-1, occurance_dim, vectorizd_occurances.shape.as_list()[-1]))
        occurance_mask = tf.reshape(occurance_len, (-1,))

        # Now (question * paragraph, dim), consistening of fixed (question, paragraph) encodings
        with tf.variable_scope("encode_occurances"):
            encoded_occurances = self.occurance_encoder.apply(is_train, vectorizd_occurances, occurance_mask)

        # Now para(question, paragraph, dim)
        encoded_occurances = tf.reshape(encoded_occurances, (batch, -1, encoded_occurances.shape.as_list()[-1]))

        with tf.variable_scope("map_paragraphs"):
            encoded_occurances = self.paragraph_mapper.apply(is_train, encoded_occurances, n_paragraphs)

        with tf.variable_scope("predict"):
            return self.prediction_layer.apply(encoded_occurances, answer , n_paragraphs)



