import unittest
import numpy as np

from docqa.data_processing.span_data import get_best_span_bounded, get_best_span_from_sent_predictions, get_best_span, \
    get_best_in_sentence_span


class TestEvaluator(unittest.TestCase):

    @staticmethod
    def best_span_brute_force(p1, p2):
        best_val = -1
        best_span = -1
        for i in range(len(p1)):
            for j in range(i, len(p2)):
                val = p1[i] * p2[j]
                if val > best_val:
                    best_val = val
                    best_span = (i, j)
        return best_span, best_val

    def test_best_span_rng(self):
        rng = np.random.RandomState(0)
        for test_num in range(0, 100):
            rng.seed(test_num)
            p1 = rng.uniform(0, 1, 15)
            p2 = rng.uniform(0, 1, 15)
            best_span, best_val = self.best_span_brute_force(p1, p2)
            pred_span, pred_val = get_best_span(p1, p2)
            self.assertEqual(best_span, pred_span)
            self.assertAlmostEqual(best_val, pred_val, 10)

    def test_best_restricted_span_rng(self):
        rng = np.random.RandomState(0)
        for test_num in range(200):
            rng.seed(test_num)
            lens = rng.random_integers(1, 4, size=3)
            p1 = [rng.uniform(0, 1, x) for x in lens]
            p2 = [rng.uniform(0, 1, x) for x in lens]
            best_span, best_val = None, -1
            offset = 0
            for i in range(len(lens)):
                span, val = self.best_span_brute_force(p1[i], p2[i])
                span = span[0] + offset, span[1] + offset
                offset += lens[i]
                if val > best_val:
                    best_span = span
                    best_val = val

            pred_span, pred_val = get_best_in_sentence_span(np.concatenate(p1), np.concatenate(p2), lens)

            self.assertEqual(best_span, pred_span)
            self.assertAlmostEqual(best_val, pred_val, 10)

    def test_best_sent_span_rng(self):
        rng = np.random.RandomState(0)
        for test_num in range(200):
            rng.seed(test_num)
            n_sent = 3
            sen_lengths = rng.random_integers(1, 15, size=n_sent)

            p1 = rng.uniform(0, 1, (n_sent, sen_lengths.max()+1))
            p2 = rng.uniform(0, 1, (n_sent, sen_lengths.max()+1))
            best_span, best_val = None, -1

            offset = 0
            for sent_ix,sent_len in enumerate(sen_lengths):
                span, val = self.best_span_brute_force(p1[sent_ix][:sent_len], p2[sent_ix][:sent_len])
                span = span[0] + offset, span[1] + offset
                offset += sent_len
                if val > best_val:
                    best_span = span
                    best_val = val

            pred_span, pred_val = get_best_span_from_sent_predictions(p1, p2, sen_lengths)

            self.assertEqual(best_span, pred_span)
            self.assertAlmostEqual(best_val, pred_val, 10)

    def test_bounded_span(self):
        p1 = np.array([0.5, 0.1, 0, 0.2])
        p2 = np.array([0.6, 0.1, 0, 0.9])

        self.assertEqual(list(get_best_span_bounded(p1, p2, 2)[0]), [0, 0])
        self.assertEqual(list(get_best_span_bounded(p1, p2, 12)[0]), [0, 3])
