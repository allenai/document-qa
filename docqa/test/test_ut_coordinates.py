import unittest
import numpy as np
import tensorflow as tf

from docqa.nn.span_prediction import to_packed_coordinates, to_unpacked_coordinates


class TestPackedCoordinates(unittest.TestCase):

    def test_random(self):
        matrix_size = 100
        sess = tf.Session()

        matrix_size_placeholder = tf.placeholder(np.int32, ())
        span_placeholder = tf.placeholder(np.int32, [None, 2])
        # tmp = tmp_lens(to_packed_coordiantes(span_placeholder, matrix_size_placeholder), matrix_size_placeholder)
        rebuilt = to_unpacked_coordinates(to_packed_coordinates(span_placeholder,
                                                                matrix_size_placeholder), matrix_size_placeholder, matrix_size)

        for i in range(0, 1000):
            rng = np.random.RandomState(i)
            n_elements = 20

            start = rng.randint(0, matrix_size-1, size=n_elements)
            end = np.zeros_like(start)
            for i in range(n_elements):
                end[i] = start[i] + rng.randint(0, matrix_size - start[i])
            spans = np.stack([start, end], axis=1)

            r = sess.run(rebuilt, {span_placeholder:spans, matrix_size_placeholder:matrix_size})
            self.assertTrue(np.all(r == spans))