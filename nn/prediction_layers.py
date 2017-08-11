from typing import List, Optional

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from configurable import Configurable
from nn.attention_recurrent_layers import WeightedWindowAttention
from nn.layers import get_keras_initialization, SequenceMapper, MergeLayer, SequenceBiMapper, Mapper, SequenceEncoder, \
    FixedMergeLayer
from nn.ops import exp_mask, VERY_NEGATIVE_NUMBER
from tensorflow import Tensor
from tensorflow.contrib.layers import fully_connected

from model import ModelOutput, Prediction


class SequencePredictionLayer(Configurable):
    """
    x=(batch, sequence, dim) -> ModelOutput
    The format of `answer` depends on the training format
    """
    def apply(self, is_train, x: Tensor, answer: List[Tensor], mask=None):
        return NotImplemented()


class AttentionPredictionLayer(Configurable):
    def apply(self, is_train, keys, context, answer: List[Tensor], mask=None, memory_mask=None):
        raise NotImplementedError()


def predict_from_bounds(answer, start_logits, end_logits, mask, aggregate):
    masked_start_logits = exp_mask(start_logits, mask)
    masked_end_logits = exp_mask(end_logits, mask)

    if len(answer) == 1:
        # answer span is encoding in a sparse int array
        answer_spans = answer[0]
        losses1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=masked_start_logits, labels=answer_spans[:, 0])
        losses2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=masked_end_logits, labels=answer_spans[:, 1])
        loss = tf.add_n([tf.reduce_mean(losses1), tf.reduce_mean(losses2)], name="loss")
    elif len(answer) == 2 and all(x.dtype == tf.bool for x in answer):
        # all correct start/end bounds are marked in a dense bool array
        # In this case there might be multiple answer spans, so we need an aggregation strategy
        losses = []
        for answer_mask, logits in zip(answer, [masked_start_logits, masked_end_logits]):
            log_norm = tf.reduce_logsumexp(logits, axis=1)
            if aggregate == "sum":
                log_score = tf.reduce_logsumexp(logits +
                                                VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer_mask, tf.float32)), axis=1)
            elif aggregate == "max":
                log_score = tf.reduce_max(logits +
                                          VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer_mask, tf.float32)), axis=1)
            else:
                raise ValueError()
            losses.append(tf.reduce_mean(-(log_score - log_norm)))
        loss = tf.add_n(losses)
    else:
        raise NotImplemented()
    return ModelOutput(loss, BoundaryPrediction(tf.nn.softmax(masked_start_logits),
                                                tf.nn.softmax(masked_end_logits),
                                                masked_start_logits, masked_end_logits))


def best_span_from_bounds(start_logits, end_logits, bound=None):
    """
    Brute force approach to finding the best span from start/end logits in tensorflow, still can
    be faster then the python dynamic-programming version
    """
    b = tf.shape(start_logits)[0]

    # Using `top_k` to get the index and value at once is faster
    # then using argmax and then gather to get in the value
    top_k = tf.nn.top_k(start_logits + end_logits, k=1)
    values, indices = [tf.squeeze(x, axis=[1]) for x in top_k]

    # Convert to (start_position, length) format
    indices = tf.stack([indices, tf.fill((b,), 0)], axis=1)

    if bound is None:
        # In this case it might best faster to just compute
        # the entire (batch x n_word x n_word) matrix
        n_lengths = tf.shape(start_logits)[1]
    else:
        # take the min in case the bound > the context
        n_lengths = tf.minimum(bound, tf.shape(start_logits)[1])

    def compute(i, values, indices):
        top_k = tf.nn.top_k(start_logits[:, :-i] + end_logits[:, i:])
        b_values, b_indices = [tf.squeeze(x, axis=[1]) for x in top_k]

        b_indices = tf.stack([b_indices, tf.fill((b, ), i)], axis=1)
        indices = tf.where(b_values > values, b_indices, indices)
        values = tf.maximum(values, b_values)
        return i+1, values, indices

    _, values, indices = tf.while_loop(
        lambda ix, values, indices: ix < n_lengths,
        compute,
        [1, values, indices],
        back_prop=False)

    spans = tf.stack([indices[:, 0], indices[:, 0] + indices[:, 1]], axis=1)
    return spans, values


# FIXME should be moved to the `span_prediction` module
class BoundaryPrediction(Prediction):
    """ Individual logits for the span start/end """
    def __init__(self, start_prob, end_prob,
                 start_logits, end_logits):
        self.start_probs = start_prob
        self.end_probs = end_prob
        self.start_logits = start_logits
        self.end_logits = end_logits
        self._bound_predictions = {}

    def get_best_span(self, bound: int):
        if bound in self._bound_predictions:
            return self._bound_predictions[bound]
        else:
            pred = best_span_from_bounds(self.start_logits, self.end_logits, bound)
            self._bound_predictions[bound] = pred
            return pred


class ConfidencePrediction(Prediction):
    """ boundary logits with an additional confidence logit """
    def __init__(self,
                 start_prob, end_prob,
                 start_logits, end_logits,
                 none_prob, non_op_logit):
        self.start_probs = start_prob
        self.end_probs = end_prob
        self.none_prob = none_prob
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.non_op_logit = non_op_logit

    def get_best_span(self, bound: int):
        return best_span_from_bounds(self.start_logits, self.end_logits, bound)


class ChainPredictor(SequencePredictionLayer):
    # FIXME should be phased out in favor of `BoundsPredictor`
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 init: str="glorot_uniform", aggregate=None):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.init = init
        self.aggregate = aggregate

    def apply(self, is_train, context_embed, answer, context_mask=None):

        init_fn = get_keras_initialization(self.init)

        with tf.variable_scope("start_layer"):
            m1 = self.start_layer.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(m1, 1, activation_fn=None,
                                      weights_initializer=init_fn)
            logits1 = tf.squeeze(logits1, squeeze_dims=[2])

        with tf.variable_scope("end_layer"):
            m2 = self.end_layer.apply(is_train, tf.concat([context_embed, m1], axis=2), context_mask)

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(m2, 1, activation_fn=None, weights_initializer=init_fn)
            logits2 = tf.squeeze(logits2, squeeze_dims=[2])

        return predict_from_bounds(answer, logits1, logits2, context_mask, self.aggregate)

    def __setstate__(self, state):
        if "aggregate" not in state["state"]:
            state["state"]["aggregate"] = None
        super().__setstate__(state)


class ChainConcat(SequenceBiMapper):
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 soft_select_start_word: bool=True, use_original: bool=True,
                 use_start_layer: bool=True, init: str="glorot_uniform"):
        self.init = init
        self.use_original = use_original
        self.start_layer = start_layer
        self.use_start_layer = use_start_layer
        self.end_layer = end_layer
        self.soft_select_start_word = soft_select_start_word

    def apply(self, is_train, context_embed, context_mask=None):
        init_fn = get_keras_initialization(self.init)
        with tf.variable_scope("start_layer"):
            m1 = self.start_layer.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(tf.concat([m1, context_embed], axis=2), 1,
                                      activation_fn=None,
                                      weights_initializer=init_fn)
            masked_logits1 = exp_mask(tf.squeeze(logits1, squeeze_dims=[2]), context_mask)
            prediction1 = tf.nn.softmax(masked_logits1)

        m2_input = []
        if self.use_original:
            m2_input.append(context_embed)
        if self.use_start_layer:
            m2_input.append(m1)
        if self.soft_select_start_word:
            soft_select = tf.einsum("ai,aik->ak", prediction1, m1)
            soft_select_tiled = tf.tile(tf.expand_dims(soft_select, axis=1), [1, tf.shape(m1)[1], 1])
            m2_input += [soft_select_tiled, soft_select_tiled * m1]

        with tf.variable_scope("end_layer"):
            m2 = self.end_layer.apply(is_train, tf.concat(m2_input, axis=2), context_mask)

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(tf.concat([m2, context_embed], axis=2), 1,
                                      activation_fn=None, weights_initializer=init_fn)

        return logits1, logits2

    def __setstate__(self, state):
        if "aggregate" not in state["state"]:
            state["state"]["aggregate"] = None
        return super().__setstate__(state)


class ChainConcatPredictor(SequencePredictionLayer):
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 soft_select_start_word: bool=True, use_original: bool=True, use_start_layer: bool=True,
                 init: str="glorot_uniform", aggregate=None):
        self.use_original = use_original
        self.start_layer = start_layer
        self.use_start_layer = use_start_layer
        self.end_layer = end_layer
        self.soft_select_start_word = soft_select_start_word
        self.init = init
        self.aggregate = aggregate

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)

        with tf.variable_scope("start_layer"):
            m1 = self.start_layer.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(tf.concat([m1, context_embed], axis=2), 1,
                                      activation_fn=None,
                                      weights_initializer=init_fn)
            logits1 = tf.squeeze(logits1, squeeze_dims=[2])
            masked_logits1 = exp_mask(logits1, context_mask)
            prediction1 = tf.nn.softmax(masked_logits1)

        m2_input = []
        if self.use_original:
            m2_input.append(context_embed)
        if self.use_start_layer:
            m2_input.append(m1)
        if self.soft_select_start_word:
            soft_select = tf.einsum("ai,aik->ak", prediction1, m1)
            soft_select_tiled = tf.tile(tf.expand_dims(soft_select, axis=1), [1, tf.shape(m1)[1], 1])
            m2_input += [soft_select_tiled, soft_select_tiled * m1]

        with tf.variable_scope("end_layer"):
            m2 = self.end_layer.apply(is_train, tf.concat(m2_input, axis=2), context_mask)

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(tf.concat([m2, context_embed], axis=2), 1,
                                      activation_fn=None, weights_initializer=init_fn)
            logits2 = tf.squeeze(logits2, squeeze_dims=[2])

        return predict_from_bounds(answer, logits1, logits2, context_mask, self.aggregate)

    def __setstate__(self, state):
        if "aggregate" not in state["state"]:
            state["state"]["aggregate"] = None
        return super().__setstate__(state)


# class AttenPredictor(SequencePredictionLayer):
#     def __init__(self, start_layer: SequenceMapper,
#                  cell: RnnCellSpec, window_size: int,
#                  merge_start_state: Optional[MergeLayer],
#                  rescale: bool, init: str="glorot_uniform"):
#         self.window_size = window_size
#         self.start_layer = start_layer
#         self.merge_start_state = merge_start_state
#         self.cell = cell
#         self.init = init
#         self.rescale = rescale
#
#     def apply(self, is_train, context_embed, answer, context_mask=None):
#         answer_start, answer_end = answer
#         init_fn = get_keras_initialization(self.init)
#
#         with tf.variable_scope("start_layer"):
#             m1 = self.start_layer.apply(is_train, context_embed, context_mask)
#
#         with tf.variable_scope("start_pred"):
#             logits1 = fully_connected(tf.concat([context_embed, m1], axis=2), 1,
#                                       activation_fn=None,
#                                       weights_initializer=init_fn)
#             logits1 = tf.squeeze(logits1, squeeze_dims=[2])
#             masked_logits1 = exp_mask(logits1, context_mask)
#             prediction1 = tf.nn.softmax(masked_logits1)
#
#         if self.merge_start_state is not None:
#             m12_in = self.merge_start_state.apply(context_embed, m1)
#         else:
#             m12_in = context_embed
#
#         with tf.variable_scope("end_layer"):
#             weight_logits = masked_logits1
#             if self.rescale:
#                 w = tf.get_variable("scale", shape=(), dtype=tf.float32, initializer=tf.ones_initializer())
#                 b = tf.get_variable("scale-bais", shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
#                 weight_logits = weight_logits*w + b
#
#             cell = WeightedWindowAttention(m1, weight_logits, self.window_size, self.cell(is_train))
#             m2 = dynamic_rnn(cell, m12_in, context_mask, dtype=tf.float32)[0]
#
#         with tf.variable_scope("end_pred"):
#             logits2 = fully_connected(m2, 1,activation_fn=None, weights_initializer=init_fn)
#             logits2 = tf.squeeze(logits2, squeeze_dims=[2])
#             masked_logits2 = exp_mask(logits2, context_mask)
#             prediction2 = tf.nn.softmax(masked_logits2)
#
#         losses1 = tf.nn.softmax_cross_entropy_with_logits(
#             logits=masked_logits1, labels=tf.cast(answer_start, 'float'))
#         losses2 = tf.nn.softmax_cross_entropy_with_logits(
#             logits=masked_logits2, labels=tf.cast(answer_end, 'float'))
#
#         loss = tf.add_n([tf.reduce_mean(losses1), tf.reduce_mean(losses2)], name="loss")
#         return ModelOutput(loss, BoundaryPrediction(prediction1, prediction2, logits1, logits2))
#
#
# class StartAttenPredictor(SequencePredictionLayer):
#     def __init__(self, start_layer: SequenceMapper,
#                  cell: RnnCellSpec, window_size: int,
#                  merge_start_state: Optional[MergeLayer],
#                  rescale: bool, init: str="glorot_uniform"):
#         self.window_size = window_size
#         self.start_layer = start_layer
#         self.merge_start_state = merge_start_state
#         self.cell = cell
#         self.init = init
#         self.rescale = rescale
#
#     def apply(self, is_train, context_embed, answer, context_mask=None):
#         answer_start, answer_end = answer
#         init_fn = get_keras_initialization(self.init)
#
#         with tf.variable_scope("start_layer"):
#             m1 = self.start_layer.apply(is_train, context_embed, context_mask)
#
#         with tf.variable_scope("start_pred"):
#             logits1 = fully_connected(tf.concat([context_embed, m1], axis=2), 1,
#                                       activation_fn=None,
#                                       weights_initializer=init_fn)
#             logits1 = tf.squeeze(logits1, squeeze_dims=[2])
#             masked_logits1 = exp_mask(logits1, context_mask)
#             prediction1 = tf.nn.softmax(masked_logits1)
#
#         if self.merge_start_state is not None:
#             m12_in = self.merge_start_state.apply(context_embed, m1)
#         else:
#             m12_in = context_embed
#
#         with tf.variable_scope("end_layer"):
#             weight_logits = masked_logits1
#             if self.rescale:
#                 w = tf.get_variable("scale", shape=(), dtype=tf.float32, initializer=tf.ones_initializer())
#                 b = tf.get_variable("scale-bais", shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
#                 weight_logits = weight_logits*w + b
#
#             cell = WeightedWindowAttention(m1, weight_logits, self.window_size, self.cell(is_train))
#             m2 = dynamic_rnn(cell, m12_in, context_mask, dtype=tf.float32)[0]
#
#         with tf.variable_scope("end_pred"):
#             logits2 = fully_connected(m2, 1,activation_fn=None, weights_initializer=init_fn)
#             logits2 = tf.squeeze(logits2, squeeze_dims=[2])
#             masked_logits2 = exp_mask(logits2, context_mask)
#             prediction2 = tf.nn.softmax(masked_logits2)
#
#         losses1 = tf.nn.softmax_cross_entropy_with_logits(
#             logits=masked_logits1, labels=tf.cast(answer_start, 'float'))
#         losses2 = tf.nn.softmax_cross_entropy_with_logits(
#             logits=masked_logits2, labels=tf.cast(answer_end, 'float'))
#
#         loss = tf.add_n([tf.reduce_mean(losses1), tf.reduce_mean(losses2)], name="loss")
#         return ModelOutput(loss, BoundaryPrediction(prediction1, prediction2, logits1, logits2))

