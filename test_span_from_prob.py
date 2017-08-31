import tensorflow as tf

from nn.span_prediction_ops import best_span_from_bounds


def main2():
    sess = tf.Session()
    l1 = tf.random_uniform((5, 4), 0, 1, seed=2)
    l2 = tf.random_uniform((5, 4), 0, 1, seed=3)
    l = tf.shape(l1)[1]
    bound = 2
    b_indices, b_values = best_span_from_bounds(l1, l2, bound)

    shaped_probs = tf.expand_dims(l1, 2) + tf.expand_dims(l2, 1)
    probs = tf.reshape(shaped_probs, (1, -1))
    # probs = tf.nn.softmax(probs)
    banded = tf.matrix_band_part(tf.reshape(probs, (-1, l, l)), 0, bound-1)

    flattended = tf.reshape(banded, (tf.shape(l1)[0], l * l))
    values, top_k_indices = tf.nn.top_k(flattended, k=1)
    values = tf.squeeze(values, 1)
    top_k_indices = tf.squeeze(top_k_indices, 1)
    indices = tf.stack([top_k_indices % l, top_k_indices // l], axis=1)

    b_i, i, top_k_indices, c1, c2, banded, flattended, l1, l2, shaped_probs = sess.run([b_indices, indices, top_k_indices,
                                                          top_k_indices // l,
                                                          top_k_indices % l,
                                                          banded, flattended, l1, l2, shaped_probs])

    print(b_i)
    print(l1)
    print(l2)
    print(shaped_probs)

    print(banded)
    print(i)
    # print(b_i)
    # print(flattended)
    # print(top_k_indices)
    # print(c1)
    # print(c2)


if __name__ == "__main__":
    main2()
