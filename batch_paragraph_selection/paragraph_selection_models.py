from typing import Optional, List, Dict

import tensorflow as tf
from paragraph_selection.selection_encoder import ParagraphQuestionGroupEncoder, ParagraphQuestionGroupFeaturizedEncoder
from tensorflow import Tensor

from data_processing.paragraph_qa import DocumentQaStats
from model import Model, ModelOutput, Prediction
from nn.embedder import WordEmbedder, CharWordEmbedder
from nn.layers import SequenceMapper, SequenceEncoder
from nn.similarity_layers import SimilarityFunction
from utils import ResourceLoader


class SelectedParagraphs(Prediction):
    def __init__(self, paragraph_probs):
        self.paragraph_probs = paragraph_probs


class ParagraphSelectionFeaturizedModel(Model):

    def __init__(self,
                 encoder: ParagraphQuestionGroupFeaturizedEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder] = None):

        if word_embed is None and char_embed is None:
            raise ValueError()
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.encoder = encoder
        self._is_train_placeholder = None

    def init(self, corpus: DocumentQaStats, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader)
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(corpus)

    def set_inputs(self, datasets: List[ParagraphSelectionDataset], word_vec_loader):
        voc = set()
        for dataset in datasets:
            voc.update(dataset.get_voc())

        self.set_input_spec(voc, word_vec_loader)

    def set_input_spec(self, voc, word_vec_loader):
        if self.word_embed is not None:
            self.word_embed.init(word_vec_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(word_vec_loader, voc)
        self.encoder.init(self.word_embed,None if self.char_embed is None else self.char_embed.embeder)
        self._is_train_placeholder = tf.placeholder(tf.bool, ())

    def get_prediction(self):
        return self.get_predictions_for(self._is_train_placeholder,
                                        {x:x for x in self.encoder.get_placeholders()})

    def get_predictions_for(self, is_train: Tensor, input_tensors: Dict[Tensor, Tensor]):
        enc = self.encoder
        q_mask = input_tensors[enc.question_len]

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

        answer = input_tensors[enc.answer_paragraph]
        return self._get_predictions_for(is_train, q_embed, q_mask, input_tensors[enc.question_features],
                                         input_tensors[enc.context_len], answer)

    def _get_predictions_for(self, is_train, question_embed, question_mask,
                             question_features, context_len, answer) -> ModelOutput:
        raise NotImplemented()

    def encode(self, batch: object, is_train: bool):
        data = self.encoder.encode(batch, is_train)
        data[self._is_train_placeholder] = is_train
        return data


class SelectedFeaturizer(ParagraphSelectionFeaturizedModel):
    def __init__(self,
                 encoder: ParagraphQuestionGroupFeaturizedEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 question_mapper: SequenceMapper,
                 normalize_context_len: bool):
        super().__init__(encoder, word_embed, char_embed)
        self.question_mapper = question_mapper
        self.normalize_context_len = normalize_context_len

    def _get_predictions_for(self, is_train, question_embed, question_mask, question_features, context_len, answer) -> ModelOutput:
        with tf.variable_scope("map_question"):
            question_rep = self.question_mapper.apply(is_train, question_embed, question_mask)

        if question_rep.shape.as_list()[-1] != question_features.shape.as_list()[-1]:
            raise ValueError("Mapper should have returned the same number of features as "
                             "featurizer, but got %d not %d" % (
                question_rep.shape.as_list()[-1], question_features.shape.as_list()[-1]))

        # (now a question x paragraph distance matrix)
        logits = tf.einsum("qwd,qwpd->qp", question_rep, question_features)

        if self.normalize_context_len:
            divisor_w = tf.get_variable("divsor_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
            divisor_rt_w = tf.get_variable("divsor_rt_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
            divisor_b = tf.get_variable("divsor_len_b", (), dtype=tf.float32, initializer=tf.ones_initializer())
            float_len = tf.cast(context_len, tf.float32)
            logits /= tf.expand_dims(float_len * divisor_w + tf.sqrt(float_len) * divisor_rt_w + divisor_b, 0)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits))
        predictions = tf.nn.softmax(logits)
        return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))


# class ProjectedSelectedFeaturizer(ParagraphSelectionFeaturizedModel):
#     def __init__(self,
#                  encoder: ParagraphQuestionGroupFeaturizedEncoder,
#                  word_embed: Optional[WordEmbedder],
#                  char_embed: Optional[CharWordEmbedder],
#                  question_mapper: SequenceMapper,
#                  reduce_layer: ReduceLayer,
#                  normalize_context_len: bool):
#         super().__init__(encoder, word_embed, char_embed)
#         self.normalize_context_len = normalize_context_len
#         self.question_mapper = question_mapper
#         self.reduce_layer = reduce_layer
#
#     def _get_predictions_for(self, is_train, question_embed, question_mask, question_features, context_len, answer) -> ModelOutput:
#         with tf.variable_scope("map_question"):
#             question_rep = self.question_mapper.apply(is_train, question_embed, question_mask)
#
#         question_w = question_rep
#
#         n_match_features = question_features.shape.as_list()[-1]
#         n_channels = question_rep.shape.as_list()[-1]//n_match_features
#
#         w_shape = tf.shape(question_w)
#         question_w = tf.reshape(question_w, (w_shape[0], w_shape[1], n_match_features, n_channels))
#         logits = tf.einsum("qwxc,qwpx->qpc", question_rep, question_features)
#
#
#         logits = self.reduce_layer.apply(is_train, logits)
#
#         if self.normalize_context_len:
#             divisor_w = tf.get_variable("divsor_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
#             divisor_rt_w = tf.get_variable("divsor_rt_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
#             divisor_b = tf.get_variable("divsor_len_b", (), dtype=tf.float32, initializer=tf.ones_initializer())
#             float_len = tf.cast(context_len, tf.float32)
#             logits /= tf.expand_dims(float_len * divisor_w + tf.sqrt(float_len) * divisor_rt_w + divisor_b, 0)
#
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits))
#         predictions = tf.nn.softmax(logits)
#         return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))


# class SelectedThenEncodeFeaturizer(ParagraphSelectionFeaturizedModel):
#     def __init__(self,
#                  encoder: ParagraphQuestionGroupFeaturizedEncoder,
#                  word_embed: Optional[WordEmbedder],
#                  char_embed: Optional[CharWordEmbedder],
#                  question_mapper: SequenceMapper,
#                 quesiton_word_encode: SequenceEncoder,
#                 question_encode_reduce: ReduceLayer,
#                  normalize_context_len):
#         super().__init__(encoder, word_embed, char_embed)
#         self.question_mapper = question_mapper
#         self.quesiton_word_encode = quesiton_word_encode
#         self.question_encode_reduce = question_encode_reduce
#         self.normalize_context_len = normalize_context_len
#
#     def _get_predictions_for(self, is_train, question_embed, question_mask, question_features, context_len, answer) -> ModelOutput:
#         with tf.variable_scope("map_question"):
#             question_rep = self.question_mapper.apply(is_train, question_embed, question_mask)
#
#         n_questions = tf.shape(question_features)[0]
#         n_words = tf.shape(question_features)[1]
#         n_para = tf.shape(question_features)[2]
#
#         # (question, para, n_words, dim)
#         question_features = tf.transpose(question_features, [0, 2, 1, 3])
#
#         # (question, para, n_words, q_features_dim + question_rep_dim)
#         features = tf.concat([question_features, tf.tile(tf.expand_dims(question_rep, 1), [1, n_para, 1, 1])], axis=3)
#         n_fe = features.shape.as_list()[3]
#
#         # (n_question * n_para, n_words, dim)
#         features = tf.reshape(features, (n_questions*n_para, n_words, n_fe))
#         tiled_mask = tf.tile(question_mask, [n_para])
#
#         with tf.variable_scope("encode_join_embed"):
#             features = self.quesiton_word_encode.apply(is_train, features, tiled_mask)
#
#         # (n_question, n_para, dim)
#         features = tf.reshape(features, (n_questions, n_para, n_fe))
#
#         # (n_question, n_para)
#         logits = self.question_encode_reduce.apply(is_train, features)
#
#         if self.normalize_context_len:
#             divisor_w = tf.get_variable("divsor_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
#             divisor_rt_w = tf.get_variable("divsor_rt_len_w", (), dtype=tf.float32, initializer=tf.zeros_initializer())
#             divisor_b = tf.get_variable("divsor_len_b", (), dtype=tf.float32, initializer=tf.ones_initializer())
#             float_len = tf.cast(context_len, tf.float32)
#             logits /= tf.expand_dims(float_len * divisor_w + tf.sqrt(float_len) * divisor_rt_w + divisor_b, 0)
#
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits))
#         predictions = tf.nn.softmax(logits)
#         return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))

#
# class SelectedAndRerank(ParagraphSelectionFeaturizedModel):
#     def __init__(self,
#                  encoder: ParagraphQuestionGroupFeaturizedEncoder,
#                  word_embed: Optional[WordEmbedder],
#                  char_embed: Optional[CharWordEmbedder],
#                  question_mapper: SequenceMapper,
#                 quesiton_word_encode: SequenceEncoder,
#                 question_encode_reduce: ReduceLayer):
#         super().__init__(encoder, word_embed, char_embed)
#         self.question_mapper = question_mapper
#         self.quesiton_word_encode = quesiton_word_encode
#         self.question_encode_reduce = question_encode_reduce
#
#     def _get_predictions_for(self, is_train, question_embed, question_mask, question_features, answer) -> ModelOutput:
#         with tf.variable_scope("map_question"):
#             question_rep = self.question_mapper.apply(is_train, question_embed, question_mask)
#
#         n_questions = tf.shape(question_features)[0]
#         n_words = tf.shape(question_features)[1]
#         n_para = tf.shape(question_features)[2]
#
#         # (question, para, n_words, dim)
#         question_features = tf.transpose(question_features, [0, 2, 1, 3])
#
#         # (question, para, n_words, q_features_dim + question_rep_dim)
#         features = tf.concat([question_features, tf.tile(tf.expand_dims(question_rep, 1), [1, n_para, 1, 1])], axis=3)
#         n_fe = features.shape.as_list()[3]
#
#         # (n_question * n_para, n_words, dim)
#         features = tf.reshape(features, (n_questions*n_para, n_words, n_fe))
#         tiled_mask = tf.tile(question_mask, [n_para])
#
#         with tf.variable_scope("encode_join_embed"):
#             features = self.quesiton_word_encode.apply(is_train, features, tiled_mask)
#
#         # (n_question, n_para, dim)
#         features = tf.reshape(features, (n_questions, n_para, n_fe))
#
#         # (n_question, n_para)
#         logits = self.question_encode_reduce.apply(is_train, features)
#
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=logits))
#         predictions = tf.nn.softmax(logits)
#         return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))


class SharedVectorModel(ParagraphSelectionModel):
    def __init__(self,
                 encoder: ParagraphQuestionGroupEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 shared_mapper: Optional[SequenceMapper],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 question_encoder: SequenceEncoder,
                 context_encoder: SequenceEncoder,
                 similiariy_fn: SimilarityFunction):
        super().__init__(encoder, word_embed, char_embed)
        self.shared_mapper = shared_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.similiariy_fn = similiariy_fn

    def _get_predictions_for(self, is_train, question_embed, question_mask, context_embed, context_mask,
                             context_sentences, answer) -> ModelOutput:
        if self.shared_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_embed = self.shared_mapper.apply(is_train, context_embed, context_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_embed = self.shared_mapper.apply(is_train, question_embed, question_mask)

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_embed = self.question_mapper.apply(is_train, question_embed, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_embed = self.context_mapper.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("encode_context"):
            context_encode = self.context_encoder.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("encode_question"):
            question_encode = self.context_encoder.apply(is_train, question_embed, question_mask)

        with tf.variable_scope("compute_similarity"):
            sim = self.similiariy_fn.get_scores(tf.expand_dims(question_encode, 0), tf.expand_dims(context_encode, 0))
            sim = tf.squeeze(sim, 0)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=sim))
        predictions = tf.nn.softmax(sim)
        return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))


# class CrossVectorModel(ParagraphSelectionModel):
#     def __init__(self,
#                  encoder: ParagraphQuestionGroupEncoder,
#                  word_embed: Optional[WordEmbedder],
#                  char_embed: Optional[CharWordEmbedder],
#                  shared_mapper: Optional[SequenceMapper],
#                  question_mapper: Optional[SequenceMapper],
#                  context_mapper: Optional[SequenceMapper],
#                  question_encoder: SequenceEncoder):
#         super().__init__(encoder, word_embed, char_embed)
#         self.shared_mapper = shared_mapper
#         self.question_mapper = question_mapper
#         self.context_mapper = context_mapper
#         self.question_encoder = question_encoder
#
#     def _get_predictions_for(self, is_train, question_embed, question_mask, context_embed, context_mask,
#                              context_sentences, answer) -> ModelOutput:
#         if self.shared_mapper is not None:
#             with tf.variable_scope("map_embed"):
#                 context_embed = self.shared_mapper.apply(is_train, context_embed, context_mask)
#             with tf.variable_scope("map_embed", reuse=True):
#                 question_embed = self.shared_mapper.apply(is_train, question_embed, question_mask)
#
#         if self.question_mapper is not None:
#             with tf.variable_scope("map_question"):
#                 question_embed = self.question_mapper.apply(is_train, question_embed, question_mask)
#
#         if self.context_mapper is not None:
#             with tf.variable_scope("map_context"):
#                 context_embed = self.context_mapper.apply(is_train, context_embed, context_mask)
#
#         with tf.variable_scope("question_encode"):
#             question_encode = self.question_encoder.apply(is_train, question_embed)  # (question, dim)
#
#         q_dim = question_encode.shape.as_list()[-1]
#         c_dim = context_embed.shape.as_list()[-1]
#
#         if q_dim != c_dim:
#             raise ValueError()
#
#         c_word_dim = tf.shape(context_embed)[1]
#         c_para_dim = tf.shape(context_embed)[0]
#
#         sent_ix = tf.cumsum(context_sentences, axis=1) - 1  # sentence end offset
#         sent_ix += tf.expand_dims(tf.range(0, c_para_dim) * c_word_dim, 1) # sentence end offset in the flattend space
#         sent_ix = tf.reshape(sent_ix, (-1,))
#
#         context_embed = tf.reshape(context_embed, (-1, c_dim))  # (para * word, dim)
#         context_encode = tf.gather(tf.reshape(context_embed, (-1, c_dim)), sent_ix)  # (para * sent, dim)
#         context_encode = tf.reshape(context_encode, (c_para_dim, -1, c_dim))    # (para, sent, dim)
#
#         sim = tf.einsum("qj,psj->qps", question_encode, context_encode)   # (question, para, sentence)
#         sim = tf.reduce_max(sim, axis=2)  # (question, para)
#
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=sim))
#         predictions = tf.nn.softmax(sim)
#         return ModelOutput(loss, SelectedParagraphs(predictions), tf.summary.scalar("loss", loss))
