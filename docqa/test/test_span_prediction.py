import unittest

import numpy as np
import tensorflow as tf
from docqa.nn.span_prediction import packed_span_f1_mask, to_unpacked_coordinates
from docqa.nn.span_prediction_ops import best_span_from_bounds
from docqa.utils import flatten_iterable

from docqa.data_processing.span_data import get_best_span_bounded, span_f1, top_disjoint_spans
from docqa.nn.ops import segment_logsumexp


class TestBestSpan(unittest.TestCase):

    def setUp(self):
        self.sess = tf.Session()

    def test_segment_log_sum_exp(self):
        sess = self.sess
        with sess.as_default():
            for i in range(10):
                groups = []
                for group_id in range(10):
                    group = []
                    for _ in range(np.random.randint(1, 5)):
                        group.append(np.random.normal(0, 2, 10))
                    groups.append(group)

                flat_groups = np.stack(flatten_iterable(groups), axis=0)
                semgents = np.array(flatten_iterable([ix]*len(g) for ix, g in enumerate(groups)))
                actual = sess.run(segment_logsumexp(flat_groups, semgents))
                expected = [np.log(np.sum(np.exp(np.concatenate(g, axis=0)))) for g in groups]
                self.assertTrue(np.allclose(actual, expected))

    def test_top_n_simple(self):
        spans, scores = top_disjoint_spans(np.array([
            1, 0, 0, 10,
            2, 2, 0, 0,
            0, 0, 3, 0,
            1, 0, 5, 4,
        ]).reshape((4, 4)), 3, 2)
        self.assertEqual(list(scores), [4, 3])
        self.assertEqual(spans.tolist(), [[3, 3], [2, 2]])

    def test_top_n_overlap(self):
        spans, scores = top_disjoint_spans(np.array([
            4, 4, 5, 4,
            0, 4, 4, 4,
            0, 0, 0, 4,
            0, 0, 0, 2,
        ]).reshape((4, 4)), 10, 5)
        self.assertEqual(list(scores), [5, 2])
        self.assertEqual(spans.tolist(), [[0, 2], [3, 3]])

    def test_best_span(self):
        bound = 5
        start_pl = tf.placeholder(tf.float32, (None, None))
        end_pl = tf.placeholder(tf.float32, (None, None))
        best_span, best_val = best_span_from_bounds(start_pl, end_pl, bound)
        sess = self.sess

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
        sess = self.sess

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