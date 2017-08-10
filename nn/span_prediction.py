from typing import List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.contrib.layers import fully_connected

from model import Prediction, ModelOutput
from nn.layers import SequenceBiMapper, MergeLayer, Mapper, get_keras_initialization, SequenceMapper, SequenceEncoder, \
    FixedMergeLayer
from nn.ops import VERY_NEGATIVE_NUMBER, exp_mask
from nn.prediction_layers import SequencePredictionLayer, BoundaryPrediction, AttentionPredictionLayer, \
    predict_from_bounds, best_span_from_bounds, ConfidencePrediction


class BoundedSpanPrediction(Prediction):
    """ Logits for each span in unpacked format (batch, word_start, distance_from_word_start), clients should
     be careful to mask out of bound values """
    def __init__(self, logits, answer):
        self.logits = logits
        self.answer = answer
        dist = logits.shape.as_list()[-1]
        flat_logits = tf.reshape(logits, (tf.shape(logits)[0], -1))
        max_span = tf.argmax(tf.reshape(logits, flat_logits), axis=1)
        start = max_span // dist

        self.predicted_span_logit = tf.gather(flat_logits, max_span)
        self.predicted_span = tf.stack([start, start + max_span % dist], axis=1)


class SpanFromBoundsPredictor(SequencePredictionLayer):
    def __init__(self, mapper: SequenceBiMapper, bound,
                 imp="py_loop",
                 init: str="glorot_uniform"):
        self.mapper = mapper
        self.imp = imp
        self.init = init
        self.bound = bound

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)

        with tf.variable_scope("predict"):
            m1, m2 = self.mapper.apply(is_train, context_embed, context_mask)

        logits1 = fully_connected(m1, 1, activation_fn=None,
                                  weights_initializer=init_fn)
        logits1 = tf.squeeze(logits1, squeeze_dims=[2])
        masked_logits1 = exp_mask(logits1, context_mask)

        logits2 = fully_connected(m2, 1, activation_fn=None, weights_initializer=init_fn)
        logits2 = tf.squeeze(logits2, squeeze_dims=[2])
        masked_logits2 = exp_mask(logits2, context_mask)

        if self.imp == "py_loop":
            all_logits = []
            batch_size = tf.shape(masked_logits1)[0]
            for i in range(self.bound):
                if i == 0:
                    all_logits.append(masked_logits1 + masked_logits2)
                else:
                    all_logits.append(tf.concat([
                        masked_logits1[:, :-i] + masked_logits2[:, i:],
                        tf.fill((batch_size, i), VERY_NEGATIVE_NUMBER)
                    ], axis=1))
            prediction = tf.concat([tf.expand_dims(x, 2) for x in all_logits], axis=2)
        else:
            raise NotImplementedError(self.imp)

        if len(answer) == 1:
            answer = answer[0]
            correct_start = answer[:, 0] * self.bound + tf.minimum(answer[:, 1]-answer[:, 0], self.bound)
            flat_predictions = tf.reshape(prediction, (tf.shape(prediction)[0], -1))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_predictions, labels=correct_start))
        else:
            raise NotImplementedError()

        return ModelOutput(loss, BoundedSpanPrediction(prediction))


def packed_span_f1_mask(spans, l, bound):
    starts = []
    ends = []
    for i in range(bound):
        s = tf.range(0, l - i, dtype=tf.int32)
        starts.append(s)
        ends.append(s + i)
    starts = tf.concat(starts, axis=0)
    ends = tf.concat(ends, axis=0)
    starts = tf.tile(tf.expand_dims(starts, 0), [tf.shape(spans)[0], 1])
    ends = tf.tile(tf.expand_dims(ends, 0), [tf.shape(spans)[0], 1])

    pred_len = tf.cast(ends - starts + 1, tf.float32)

    span_start = tf.maximum(starts, spans[:, 0:1])
    span_stop = tf.minimum(ends, spans[:, 1:2])

    overlap_len = tf.cast(span_stop - span_start + 1, tf.float32)
    true_len = tf.cast(spans[:, 1:2] - spans[:, 0:1] + 1, tf.float32)

    p = overlap_len / pred_len
    r = overlap_len / true_len
    return tf.where(overlap_len > 0, 2 * p * r / (p + r), tf.zeros(tf.shape(starts)))


def to_packed_coordinates(spans, l, bound=None):
    """ Converts the spans to vector of packed coordiantes, in the packed format
    spans are indexed first by length, then by start position. If bound is given
     spans are truncated to be of `bound` length """
    lens = spans[:, 1] - spans[:, 0]
    if bound is not None:
        lens = tf.minimum(lens, bound-1)
    return spans[:, 0] + l * lens - lens * (lens - 1) // 2


def to_packed_coordinates_np(spans, l, bound=None):
    """ Converts the spans to vector of packed coordiantes, in the packed format
    spans are indexed first by length, then by start position. If bound is given
     spans are truncated to be of `bound` length """
    lens = spans[:, 1] - spans[:, 0]
    if bound is not None:
        lens = np.minimum(lens, bound-1)
    return spans[:, 0] + l * lens - lens * (lens - 1) // 2


def to_unpacked_coordinates(ix, l, bound):
    ix = tf.cast(ix, tf.int32)
    # I think you can actually compute the lens in closed form:
    # lens = tf.floor(0.5 * (-tf.sqrt(4 * tf.square(l) + 4 * l - 8 * ix + 1) + 2 * l + 1))
    # but it is very ugly and rounding errors could cause problems, so this approach seems safer
    lens = []
    for i in range(bound):
        lens.append(tf.fill((l - i,), i))
    lens = tf.concat(lens, axis=0)
    lens = tf.gather(lens, ix)
    answer_start = ix - l * lens + lens * (lens - 1) // 2
    return tf.stack([answer_start, answer_start+lens], axis=1)


class PackedSpanPrediction(Prediction):
    """ Logits for each span in packed format (batch, packed_coordinate) """
    def __init__(self, logits, l, bound):
        self.bound = bound
        self.logits = logits
        argmax = tf.argmax(logits, axis=1)
        self.best_score = tf.reduce_max(logits, axis=1)
        self.predicted_span = to_unpacked_coordinates(argmax, l, bound)

    def get_best_span(self, bound):
        if bound != self.bound:
            # In theory we could easily support bounds lower then `self.bound`
            raise ValueError()
        return self.predicted_span, self.best_score


def predict_span_from_bound_logits(logits1, logits2, mask, answer, bound, aggregate):
    masked_logits1 = exp_mask(logits1, mask)
    masked_logits2 = exp_mask(logits2, mask)

    span_logits = []
    for i in range(bound):
        if i == 0:
            span_logits.append(masked_logits1 + masked_logits2)
        else:
            span_logits.append(masked_logits1[:, :-i] + masked_logits2[:, i:])
    span_logits = tf.concat(span_logits, axis=1)

    if len(answer) == 1:
        answer = answer[0]
        if answer.dtype == tf.int32:
            answer_ix = to_packed_coordinates(answer, tf.shape(logits1)[1], bound)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=span_logits, labels=answer_ix))
        else:
            log_norm = tf.reduce_logsumexp(span_logits, axis=1)
            if aggregate == "sum":
                log_score = tf.reduce_logsumexp(span_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer, tf.float32)),
                                                axis=1)
            elif aggregate == "max":
                log_score = tf.reduce_max(span_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer, tf.float32)),
                                          axis=1)
            else:
                raise NotImplementedError()
            loss = tf.reduce_mean(-(log_score - log_norm))
    else:
        raise NotImplementedError()
    return ModelOutput(loss, PackedSpanPrediction(span_logits, tf.shape(logits1)[1], bound))


class SpanFromBoundsPacked(SequencePredictionLayer):
    def __init__(self, mapper: SequenceBiMapper, bound,
                 imp="py_loop",
                 init: str="glorot_uniform",
                 aggregate="sum"):
        self.mapper = mapper
        self.imp = imp
        self.init = init
        self.bound = bound
        self.aggregate = aggregate

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)

        with tf.variable_scope("predict"):
            m1, m2 = self.mapper.apply(is_train, context_embed, context_mask)

        logits1 = fully_connected(m1, 1, activation_fn=None,
                                  weights_initializer=init_fn)
        logits1 = tf.squeeze(logits1, squeeze_dims=[2])

        logits2 = fully_connected(m2, 1, activation_fn=None, weights_initializer=init_fn)
        logits2 = tf.squeeze(logits2, squeeze_dims=[2])

        return predict_span_from_bound_logits(logits1, logits2, context_mask, answer, self.bound, self.aggregate)


class ChainConcatSpanPredictor(SequencePredictionLayer):
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper, bound: int,
                 soft_select_start_word: bool=True, use_original: bool=True, use_start_layer: bool=True,
                 init: str="glorot_uniform", aggregate=None):
        self.bound = bound
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

        return predict_span_from_bound_logits(logits1, logits2, context_mask, answer, self.bound, self.aggregate)


class SpanFromVectorBound(SequencePredictionLayer):
    def __init__(self,
                 mapper: SequenceBiMapper,
                 pre_process: Optional[SequenceMapper],
                 merge: MergeLayer,
                 post_process: Mapper,
                 bound,
                 linear_logits=False,
                 imp="py_loop",
                 f1_weight=0,
                 init: str="glorot_uniform"):
        self.mapper = mapper
        self.linear_logits = linear_logits
        self.pre_process = pre_process
        self.merge = merge
        self.post_process = post_process
        self.imp = imp
        self.init = init
        self.f1_weight = f1_weight
        self.bound = bound

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)
        bool_mask = tf.sequence_mask(context_mask, tf.shape(context_embed)[1])

        with tf.variable_scope("predict"):
            m1, m2 = self.mapper.apply(is_train, context_embed, context_mask)

        if self.pre_process is not None:
            with tf.variable_scope("pre-process1"):
                m1 = self.pre_process.apply(is_train, m1, context_mask)
            with tf.variable_scope("pre-process2"):
                m2 = self.pre_process.apply(is_train, m2, context_mask)

        if self.imp == "py_loop":
            span_vector_lst = []
            mask_lst = []
            for i in range(self.bound):
                if i == 0:
                    with tf.variable_scope("merge"):
                        span_vector_lst.append(self.merge.apply(is_train, m1, m2))
                    mask_lst.append(bool_mask)
                else:
                    with tf.variable_scope("merge", reuse=True):
                        span_vector_lst.append(self.merge.apply(is_train, m1[:, :-i], m2[:, i:]))
                    mask_lst.append(bool_mask[:, i:])

            mask = tf.concat(mask_lst, axis=1)
            span_vectors = tf.concat(span_vector_lst, axis=1)  # all logits -> flattened per-span predictions
        elif self.imp == "transpose_map":
            raise NotImplementedError()
        else:
            raise NotImplementedError(self.imp)

        span_vectors = self.post_process.apply(is_train, span_vectors)
        if not self.linear_logits:
            logits = fully_connected(span_vectors, 1, activation_fn=None, weights_initializer=init_fn)
        else:
            logits = span_vectors
        logits = tf.squeeze(logits, squeeze_dims=[2])
        logits = logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(tf.concat(mask, axis=1), tf.float32))

        l = tf.shape(context_embed)[1]

        if len(answer) == 1:
            answer = answer[0]
            if self.f1_weight == 0:
                answer_ix = to_packed_coordinates(answer, l, self.bound)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answer_ix))
            else:
                f1_mask = packed_span_f1_mask(answer, l, self.bound)
                if self.f1_weight < 1:
                    f1_mask *= self.f1_weight
                    f1_mask += (1-self.f1_weight) * tf.one_hot(to_packed_coordinates(answer, l, self.bound), l)

                # TODO can we stay in log space?  (actually its tricky since f1_mask can have zeros...)
                probs = tf.nn.softmax(logits)
                loss = -tf.reduce_mean(tf.log(tf.reduce_sum(probs * f1_mask, axis=1)))
        else:
            raise NotImplementedError()

        return ModelOutput(loss, PackedSpanPrediction(logits, l, self.bound))


class BoundsPredictor(SequencePredictionLayer):
    def __init__(self, predictor: SequenceBiMapper, init: str="glorot_uniform",
                 aggregate=None):
        self.predictor = predictor
        self.init = init
        self.aggregate = aggregate

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)
        with tf.variable_scope("bounds_encoding"):
            m1, m2 = self.predictor.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(m1, 1, activation_fn=None,
                                      weights_initializer=init_fn)
            logits1 = tf.squeeze(logits1, squeeze_dims=[2])

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(m2, 1, activation_fn=None, weights_initializer=init_fn)
            logits2 = tf.squeeze(logits2, squeeze_dims=[2])

        return predict_from_bounds(answer, logits1, logits2, context_mask, self.aggregate)


class ConfidencePredictor(SequencePredictionLayer):
    """
    Optimize log probabilty of picking the correct span, or selecting a no-op, where
    the probability is P(op)P(start)P(end) if a span exists otherwise 1 - P(nop)
    This reduces op_logit + start_logit + end_logit (where an answer exists) -
        1 - op_logit (where answer exists)
    """
    def __init__(self,
                 predictor: SequenceBiMapper,
                 encoder: SequenceEncoder,
                 confidence_predictor: Mapper,
                 init: str="glorot_uniform",
                 aggregate=None):
        self.predictor = predictor
        self.init = init
        self.aggregate = aggregate
        self.confidence_predictor = confidence_predictor
        self.encoder = encoder

    def apply(self, is_train, context_embed, answer, context_mask=None):
        init_fn = get_keras_initialization(self.init)
        m1, m2 = self.predictor.apply(is_train, context_embed, context_mask)

        if m1.shape.as_list()[-1] != 1:
            with tf.variable_scope("start_pred"):
                start_logits = fully_connected(m1, 1, activation_fn=None,
                                          weights_initializer=init_fn)
        else:
            start_logits = m1
        start_logits = tf.squeeze(start_logits, squeeze_dims=[2])

        if m1.shape.as_list()[-1] != 1:
            with tf.variable_scope("end_pred"):
                end_logits = fully_connected(m2, 1, activation_fn=None, weights_initializer=init_fn)
        else:
            end_logits = m2
        end_logits = tf.squeeze(end_logits, squeeze_dims=[2])

        masked_start_logits = exp_mask(start_logits, context_mask)
        masked_end_logits = exp_mask(end_logits, context_mask)

        start_atten = tf.einsum("ajk,aj->ak", m1, tf.nn.softmax(masked_start_logits))
        end_atten = tf.einsum("ajk,aj->ak", m2, tf.nn.softmax(masked_end_logits))
        with tf.variable_scope("encode_context"):
            enc = self.encoder.apply(is_train, context_embed, context_mask)
        with tf.variable_scope("confidence"):
            none_logit = self.confidence_predictor.apply(is_train, tf.concat([start_atten, end_atten, enc],axis=1))
        with tf.variable_scope("confidence_logits"):
            none_logit = fully_connected(none_logit, 1, activation_fn=None,
                                   weights_initializer=init_fn)
            none_logit = tf.squeeze(none_logit, axis=1)

        batch_dim = tf.shape(start_logits)[0]

        # (batch, (l * l)) logits for each (start, end) pair
        all_logits = tf.reshape(tf.expand_dims(start_logits, 1) +
                                tf.expand_dims(end_logits, 2),
                                (batch_dim, -1))

        # (batch, (l * l) + 1) logits including the none option
        all_logits = tf.concat([all_logits, tf.expand_dims(none_logit, 1)], axis=1)
        log_norms = tf.reduce_logsumexp(all_logits, axis=1)

        # Now build a "correctness" mask in the same format
        correct_mask = tf.logical_and(tf.expand_dims(answer[0], 1), tf.expand_dims(answer[1], 2))
        correct_mask = tf.reshape(correct_mask, (batch_dim, -1))
        correct_mask = tf.concat([correct_mask, tf.logical_not(tf.reduce_any(answer[0], axis=1, keep_dims=True))], axis=1)

        log_correct = tf.reduce_logsumexp(all_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(correct_mask, tf.float32)), axis=1)
        loss = tf.reduce_mean(-(log_correct - log_norms))
        probs = tf.nn.softmax(all_logits)
        return ModelOutput(loss, ConfidencePrediction(None, None, masked_start_logits, masked_end_logits,
                                                      probs[:, -1], none_logit))

        # max_logit = tf.reduce_max(masked_start_logits, axis=1) +\
        #             tf.reduce_max(masked_end_logits, axis=1)
        # max_logit = tf.maximum(max_logit, none_logit)
        # max_logit_ex = tf.expand_dims(max_logit, 1)
        # norm = tf.reduce_sum(tf.exp(masked_end_logits - max_logit_ex), axis=1) * \
        #        tf.reduce_sum(tf.exp(masked_start_logits - max_logit_ex), axis=1) +\
        #        tf.exp(none_logit - max_logit)
        # # print(max_logit.shape)
        # # print(tf.reduce_sum(tf.exp(masked_end_logits - max_logit), axis=1).shape)
        # # print(tf.exp(none_logit - max_logit).shape)
        # norm = tf.log(norm) + max_logit
        # if len(norm.shape) > 1:
        #     raise ValueError()
        #
        # score_starts = start_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer[0], tf.float32))
        # score_ends = end_logits + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer[1], tf.float32))
        # score_nones = none_logit + VERY_NEGATIVE_NUMBER * tf.cast(tf.reduce_any(answer[0], axis=1), tf.float32)
        # max_logit = tf.reduce_max(score_starts, axis=1) +\
        #             tf.reduce_max(score_ends, axis=1)
        # max_logit = tf.maximum(max_logit, score_nones)
        # max_logit_ex = tf.expand_dims(max_logit, 1)
        # score = tf.reduce_sum(tf.exp(score_starts - max_logit_ex), axis=1) *  \
        #        tf.reduce_sum(tf.exp(score_ends - max_logit_ex), axis=1) +\
        #        tf.exp(score_nones - max_logit)
        # score = tf.log(score) + max_logit
        #
        # loss = tf.reduce_mean(-(score - norm))
        # return ModelOutput(loss, ConfidencePrediction(None, None,
        #                                               masked_start_logits, masked_end_logits, none_logit))


class WithFixedContextPredictionLayer(AttentionPredictionLayer):
    def __init__(self, context_mapper: SequenceMapper, context_encoder: SequenceEncoder,
                 merge: FixedMergeLayer, bounds_predictor: SequenceBiMapper,
                 init: str = "glorot_uniform", aggregate="sum"):
        self.context_mapper = context_mapper
        self.context_encoder = context_encoder
        self.bounds_predictor = bounds_predictor
        self.merge = merge
        self.aggregate = aggregate
        self.init = init

    def apply(self, is_train, x, memories, answer: List[Tensor], x_mask=None, memory_mask=None):
        with tf.variable_scope("map_context"):
            memories = self.context_mapper.apply(is_train, memories, memory_mask)
        with tf.variable_scope("encode_context"):
            encoded = self.context_encoder.apply(is_train, memories, memory_mask)
        with tf.variable_scope("merge"):
            x = self.merge.apply(is_train, x, encoded, x_mask)
        with tf.variable_scope("predict"):
            m1, m2 = self.bounds_predictor.apply(is_train, x, x_mask)

        init = get_keras_initialization(self.init)
        with tf.variable_scope("logits1"):
            l1 = fully_connected(m1, 1, activation_fn=None, weights_initializer=init)
            l1 = tf.squeeze(l1, squeeze_dims=[2])
        with tf.variable_scope("logits2"):
            l2 = fully_connected(m2, 1, activation_fn=None, weights_initializer=init)
            l2 = tf.squeeze(l2, squeeze_dims=[2])

        return predict_from_bounds(answer, l1, l2, x_mask, self.aggregate)

