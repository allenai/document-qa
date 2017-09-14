import tensorflow as tf

from nn.layers import get_keras_initialization, SequenceMapper, SequencePredictionLayer
from nn.ops import exp_mask, VERY_NEGATIVE_NUMBER
from tensorflow.contrib.layers import fully_connected

from nn.span_prediction import BoundaryPrediction, ConfidencePrediction


class ChainPredictor(SequencePredictionLayer):
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 init: str="glorot_uniform",
                 aggregate=None):
        raise ValueError("Deprecated")
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

        with tf.variable_scope("predict_from_bounds"):
            return self.bound_predictor.predict(answer, logits1, logits2, context_mask)

    def __setstate__(self, state):
        if "aggregate" in state["state"]:
            state["state"]["aggregate"] = None
        super().__setstate__(state)


class ChainConcatPredictor(SequencePredictionLayer):
    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 soft_select_start_word: bool=True, use_original: bool=True, use_start_layer: bool=True,
                 init: str="glorot_uniform", aggregate=None):
        raise ValueError("Deprecated")
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

        # raise NotImplementedError()
        return predict_from_bounds(answer, logits1, logits2, context_mask, self.aggregate)

    def __setstate__(self, state):
        if "aggregate" not in state["state"]:
            state["state"]["aggregate"] = None
        return super().__setstate__(state)


class ChainConcatConfidenceHack(SequencePredictionLayer):
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
            masked_start_logits = exp_mask(logits1, context_mask)
            prediction1 = tf.nn.softmax(masked_start_logits)

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
            masked_end_logits = fully_connected(tf.concat([m2, context_embed], axis=2), 1,
                                      activation_fn=None, weights_initializer=init_fn)
            masked_end_logits = tf.squeeze(masked_end_logits, squeeze_dims=[2])

        batch_dim = tf.shape(masked_start_logits)[0]
        none_logit = tf.get_variable("none-logit", initializer=-1.0)
        tf.get_default_session().run(none_logit.initializer)
        none_logit = tf.tile(tf.expand_dims(none_logit, 0), [tf.shape(context_embed)[0]])


        # (batch, (l * l)) logits for each (start, end) pair

        all_logits = tf.reshape(tf.expand_dims(masked_start_logits, 1) +
                                tf.expand_dims(masked_end_logits, 2),
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
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return ConfidencePrediction(probs[:, :-1], masked_start_logits, masked_end_logits,
                                    probs[:, -1], none_logit)

    def __setstate__(self, state):
        if "aggregate" not in state["state"]:
            state["state"]["aggregate"] = None
        return super().__setstate__(state)


def predict_from_bounds(answer, start_logits, end_logits, mask, aggregate):
    masked_start_logits = exp_mask(start_logits, mask)
    masked_end_logits = exp_mask(end_logits, mask)

    if len(answer) == 1:
        answer_spans = answer[0]
        losses1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=masked_start_logits, labels=answer_spans[:, 0])
        losses2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=masked_end_logits, labels=answer_spans[:, 1])
        loss = tf.add_n([tf.reduce_mean(losses1), tf.reduce_mean(losses2)], name="loss")
    elif len(answer) == 2 and all(x.dtype == tf.bool for x in answer):
        # In this case there might be multiple answer spans, so we need an aggregation strategy
        if aggregate is None:
            raise ValueError()
        losses = []
        # for i, l in enumerate([logits1, logits2]):
        #     l = tf.exp(l) * tf.cast(context_mask, tf.float32)
        #     norm = tf.reduce_sum(l, axis=1)
        #     score = tf.reduce_sum(l * tf.cast(answer[i], tf.float32), axis=1)
        #     losses.append(tf.reduce_mean(-(tf.log(score) - tf.log(norm))))
        for i, l in enumerate([masked_start_logits, masked_end_logits]):
            log_norm = tf.reduce_logsumexp(l, axis=1)
            if aggregate is None or aggregate == "sum":
                log_score = tf.reduce_logsumexp(l + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer[i], tf.float32)), axis=1)
            elif aggregate == "max":
                log_score = tf.reduce_max(l + VERY_NEGATIVE_NUMBER * (1 - tf.cast(answer[i], tf.float32)), axis=1)
            else:
                raise ValueError()
            losses.append(tf.reduce_mean(-(log_score - log_norm)))
        loss = tf.add_n(losses)
    else:
        raise NotImplemented()

    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return BoundaryPrediction(tf.nn.softmax(masked_start_logits),
                              tf.nn.softmax(masked_end_logits),
                              masked_start_logits, masked_end_logits)
