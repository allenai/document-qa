from typing import List, Optional, Dict, Union

import tensorflow as tf
from tensorflow import Tensor

from configurable import Configurable
from experimental.mc_encoder import McQuestionEncoder
from model import ModelOutput, Model, Prediction
from nn.embedder import WordEmbedder, CharWordEmbedder
from nn.layers import SequenceMapper, SequenceBiMapper, AttentionMapper, SequenceEncoder, \
    SequencePredictionLayer, AttentionPredictionLayer
from nn.similarity_layers import SimilarityFunction
from utils import ResourceLoader, max_or_none


class McModel(Model):

    def __init__(self,
                 encoder: McQuestionEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder] = None):

        if word_embed is None and char_embed is None:
            raise ValueError()
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.encoder = encoder
        self._is_train_placeholder = None

    def init(self, corpus, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader, corpus.special_tokens)
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(corpus)

    def set_inputs(self, datasets: List, word_vec_loader):
        voc = set()
        for dataset in datasets:
            voc.update(dataset.get_vocab())

        batch_size = datasets[0].get_fixed_batch_size()
        for dataset in datasets[1:]:
            batch_size = max_or_none(dataset.get_fixed_batch_size(), batch_size)

        return self.set_input_spec(batch_size, voc, word_vec_loader)

    def set_input_spec(self, batch_size, voc, word_vec_loader):
        if self.word_embed is not None:
            self.word_embed.init(word_vec_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(word_vec_loader, voc)
        self.encoder.init(batch_size, True, self.word_embed,
                          None if self.char_embed is None else self.char_embed.embeder)
        self._is_train_placeholder = tf.placeholder(tf.bool, ())
        return self.encoder.get_placeholders()

    def get_placeholders(self):
        return self.encoder.get_placeholders() + [self._is_train_placeholder]

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]):
        is_train = input_tensors[self._is_train_placeholder]
        enc = self.encoder
        q_mask = input_tensors[enc.question_len]
        c_mask = input_tensors[enc.context_len]
        a_mask = input_tensors[enc.answer_len]

        q_embed = []
        c_embed = []
        a_embed = []

        if enc.question_chars in input_tensors:
            with tf.variable_scope("char-embed"):
                q, c, a = self.char_embed.embed(is_train,
                                                (input_tensors[enc.question_chars], q_mask),
                                                (input_tensors[enc.context_chars], c_mask),
                                                (input_tensors[enc.answer_chars], a_mask))
            q_embed.append(q)
            c_embed.append(c)
            a_embed.append(a)

        if enc.question_words in input_tensors:
            with tf.variable_scope("word-embed"):
                q, c, a = self.word_embed.embed(is_train,
                                                (input_tensors[enc.question_words], q_mask),
                                                (input_tensors[enc.context_words], c_mask),
                                                (input_tensors[enc.answer_words], a_mask))
            q_embed.append(q)
            c_embed.append(c)
            a_embed.append(a)

        q_embed = tf.concat(q_embed, axis=2)
        c_embed = tf.concat(c_embed, axis=2)
        a_embed = tf.concat(a_embed, axis=3)
        return self._get_predictions_for(is_train,
                                         q_embed, q_mask,
                                         c_embed, c_mask,
                                         a_embed, a_mask,
                                         input_tensors[enc.answer])

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer_embed, answer_mask,
                             answer) -> ModelOutput:
        raise NotImplemented()

    def encode(self, batch: List, is_train: bool):
        data = self.encoder.encode(batch, is_train)
        data[self._is_train_placeholder] = is_train
        return data


class McPrediction(Prediction):
    def __init__(self, answer_logits):
        self.answer_logits = answer_logits


class McPredictionLayer(Configurable):
    def apply(self, is_train, context_rep, answer_rep, context_mask, answer_mask, answer) -> ModelOutput:
        raise NotImplementedError()


class EncodedAnswerAttention(McPredictionLayer):
    def __init__(self, answer_encoder: SequenceEncoder, sim: SimilarityFunction,
                 reduce_similarity: SequenceEncoder):
        self.answer_encoder = answer_encoder
        self.sim = sim
        self.reduce_similarity = reduce_similarity

    def apply(self, is_train, context_rep, answer_rep, context_mask, answer_mask, answer) -> ModelOutput:
        n_options = answer_rep.shape.as_list()[1]

        with tf.variable_scope("encoder_answer_options"):
            answer_mask = tf.reshape(answer_mask, (-1,))
            answer_rep = tf.reshape(answer_rep, (-1, tf.shape(answer_rep)[2], answer_rep.shape.as_list()[-1]))
            answer_rep = self.answer_encoder.apply(is_train, answer_rep, answer_mask)
            answer_rep = tf.reshape(answer_rep, (-1, n_options, answer_rep.shape.as_list()[-1]))

        with tf.variable_scope("similarity"):
            # (batch, context_word, answer_option) scores
            sim = self.sim.get_scores(context_rep, answer_rep)

        with tf.variable_scope("reduce"):
            # (batch, answer_option)
            answer_scores = self.reduce_similarity.apply(is_train, sim, context_mask)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=answer, logits=answer_scores)
        return ModelOutput(loss, McPrediction(answer_scores))


class McAttention(McModel):
    def __init__(self, encoder: McQuestionEncoder,
                 word_embed: WordEmbedder,
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 predictor: Union[SequencePredictionLayer, AttentionPredictionLayer]):
        super().__init__(encoder, word_embed, char_embed)
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.predictor = predictor

    def _get_predictions_for(self, is_train,
                             question_rep, question_mask,
                             context_rep, context_mask,
                             answer_rep, answer_mask,
                             answer) -> ModelOutput:
        flat_answer_mask = tf.reshape(answer_mask, (-1,))
        batch_dim = tf.shape(question_rep)[0]

        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)
                answer_rep = tf.reshape(answer, (-1, tf.shape(answer_rep)[2], answer_rep.shape.as_list()[-1]))
                answer_rep = self.embed_mapper.apply(is_train, answer_rep, flat_answer_mask)
                answer_rep = tf.reshape(answer, (batch_dim, self.encoder.n_options, -1, answer_rep.shape.as_list()[-1]))

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("buid_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.match_encoder is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.match_encoder.apply(is_train, context_rep, context_mask)
