import tensorflow as tf
import numpy as np

VERY_NEGATIVE_NUMBER = -1e30


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)


# def sparse_logsumexp(xs, mask, axis, keep_dims=False):
#     my_max = tf.stop_gradient(tf.reduce_max(xs, axis=axis, keep_dims=True))
#     result = tf.log(tf.reduce_sum(tf.exp(xs - my_max), axis, keep_dims=True)) + my_max
#     if not keep_dims:
#       if isinstance(axis, int):
#         axis = [axis]
#       result = tf.squeeze(result, axis)
#     return repr()


def mixed_dropout(is_train, x, keep_probs, seed=None):
    """ dropout `x` depending on the last dimension of x, `keep_probs` """
    with tf.name_scope("dropout"):
        def drop():
            random_tensor = tf.random_uniform(tf.shape(x), seed=seed, dtype=x.dtype)
            broadcast_probs = keep_probs
            for i in range(len(x.shape)-1):
                broadcast_probs = tf.expand_dims(broadcast_probs, 0)
            random_tensor = random_tensor + broadcast_probs

            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_tensor = tf.floor(random_tensor)
            ret = tf.div(x, broadcast_probs) * binary_tensor
            ret.set_shape(x.get_shape())
            return ret

        return tf.cond(is_train, drop, lambda: x)

#
# def reduce_slices(matrix, slices):
#     """
#     Reduces the matrix by performing max(matrix[:, slice[i-1]:slice[i]])
#     """
#
#     counter = tf.constant(1, dtype=np.int32)
#     results = tf.TensorArray(dtype=tf.float32, tf.)
#
#     tf.while_loop()


def mask_entries(val, mask):
    seq_len_mask = tf.sequence_mask(
        mask,
        maxlen=tf.shape(val)[1],
        dtype=val.dtype)
    seq_len_mask = tf.expand_dims(seq_len_mask, 2)
    return val * seq_len_mask


def exp_mask(val, mask):
    return val + (1 - tf.cast(tf.sequence_mask(mask, tf.shape(val)[1]), 'float')) * VERY_NEGATIVE_NUMBER


