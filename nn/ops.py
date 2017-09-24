import tensorflow as tf

VERY_NEGATIVE_NUMBER = -1e29


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)


def segment_logsumexp(xs, segments):
    """ Similar tf.segment_sum but compute logsumexp rather then sum """
    # Stop gradients following the implementation of tf.reduce_logsumexp
    maxs = tf.stop_gradient(tf.reduce_max(xs, axis=1))
    segment_maxes = tf.segment_max(maxs, segments)
    xs -= tf.expand_dims(tf.gather(segment_maxes, segments), 1)
    sums = tf.reduce_sum(tf.exp(xs), axis=1)
    return tf.log(tf.segment_sum(sums, segments)) + segment_maxes


def exp_mask(val, mask):
    mask = tf.cast(tf.sequence_mask(mask, tf.shape(val)[1]), tf.float32)
    return val * mask + (1 - mask) * VERY_NEGATIVE_NUMBER


