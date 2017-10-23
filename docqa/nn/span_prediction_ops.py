import tensorflow as tf
import numpy as np


"""
Some utility functions for dealing with span prediction in tensorflow
"""


def best_span_from_bounds(start_logits, end_logits, bound=None):
    """
    Brute force approach to finding the best span from start/end logits in tensorflow, still usually
    faster then the python dynamic-programming version
    """
    b = tf.shape(start_logits)[0]

    # Using `top_k` to get the index and value at once is faster
    # then using argmax and then gather to get in the value
    top_k = tf.nn.top_k(start_logits + end_logits, k=1)
    values, indices = [tf.squeeze(x, axis=[1]) for x in top_k]

    # Convert to (start_position, length) format
    indices = tf.stack([indices, tf.fill((b,), 0)], axis=1)

    # TODO Might be better to build the batch x n_word x n_word
    # matrix and use tf.matrix_band to zero out the unwanted ones...

    if bound is None:
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
    spans are indexed first by length, then by start position in a flattened array.
    If bound is given spans are truncated to be of `bound` length """
    lens = spans[:, 1] - spans[:, 0]
    if bound is not None:
        lens = np.minimum(lens, bound-1)
    return spans[:, 0] + l * lens - lens * (lens - 1) // 2


def to_unpacked_coordinates(ix, l, bound):
    ix = tf.cast(ix, tf.int32)
    # You can actually compute the lens in closed form:
    # lens = tf.floor(0.5 * (-tf.sqrt(4 * tf.square(l) + 4 * l - 8 * ix + 1) + 2 * l + 1))
    # but it is very ugly and rounding errors could cause problems, so this approach seems safer
    lens = []
    for i in range(bound):
        lens.append(tf.fill((l - i,), i))
    lens = tf.concat(lens, axis=0)
    lens = tf.gather(lens, ix)
    answer_start = ix - l * lens + lens * (lens - 1) // 2
    return tf.stack([answer_start, answer_start+lens], axis=1)

