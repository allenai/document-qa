import unittest

import numpy as np
import tensorflow as tf

from data_processing.span_data import get_best_span_bounded, span_f1
from nn.prediction_layers import best_span_from_bounds
from nn.span_prediction import packed_span_f1_mask, to_unpacked_coordinates


class TestBestSpan(unittest.TestCase):

    def test_best_span(self):
        bound = 5
        start_pl = tf.placeholder(tf.float32, (None, None))
        end_pl = tf.placeholder(tf.float32, (None, None))
        best_span, best_val = best_span_from_bounds(start_pl, end_pl, bound)
        sess = tf.Session()

        for i in range(0, 20):
            rng = np.random.RandomState(i)
            l = rng.randint(50, 200)
            batch = rng.randint(1, 60)

            start = rng.uniform(size=(batch, l))
            end = rng.uniform(size=(batch, l))

            # exp since the tf version uses logits and the py version use probabilities
            expected_span, expected_score = zip(*[get_best_span_bounded(np.exp(start[i]), np.exp(end[i]), bound)
                                                  for i in range(batch)])

            actual_span, actuals_score = sess.run([best_span, best_val], {start_pl:start, end_pl:end})

            self.assertTrue(np.all(np.array(expected_span) == actual_span))
            self.assertTrue(np.allclose(expected_score, np.exp(actuals_score)))

    def test_span_f1(self):
        bound = 15
        batch_size = 5
        l = 20

        spans_pl = tf.placeholder(tf.int32, (None, 2))
        coordinate_pl = tf.placeholder(tf.int32, (1,))

        mask = packed_span_f1_mask(spans_pl, 15, bound)
        coordinates = to_unpacked_coordinates(coordinate_pl, 15, bound)[0]
        sess = tf.Session()

        for i in range(0, 20):
            rng = np.random.RandomState(i)
            starts = rng.randint(0, l, batch_size)
            ends = [rng.randint(0, l-x) + x for x in starts]
            spans = np.stack([starts, np.array(ends)], axis=1)

            f1_mask = sess.run(mask, {spans_pl:spans})

            for i in range(batch_size):
                coord = np.random.randint(0, f1_mask.shape[1])
                x,y = sess.run(coordinates, {coordinate_pl:[coord]})
                expected = span_f1(spans[i], (x, y))
                actual = f1_mask[i, coord]
                self.assertAlmostEqual(expected, actual, places=5)