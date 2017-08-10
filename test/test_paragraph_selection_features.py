import unittest

import numpy as np

from data_processing.document_splitter import ExtractedParagraph
from data_processing.text_utils import WordNormalizer
from paragraph_selection.paragraph_selection_featurizer import WordMatchingFeaturizer, ExtractedParagraph, \
    WhatWordMatchingFeaturizer, NGramMatchingFeaturizer


class MockNormalizer(object):
    def normalize(self, word):
        return word


class MockStopwords(object):
    @property
    def words(self):
        return ["Where"]


class TestSelectionFeatures(unittest.TestCase):

    def test_matching_features(self):
        fe = NGramMatchingFeaturizer(MockStopwords(), WordNormalizer(), (1, 2))
        features = fe.get_joint_features(
            ["Where", "cat", "Bill", "dog", "Pen", "dog"],
            [ExtractedParagraph(x, 0, 0, None) for x in [
                [["the", "fox"]],
                [["the", "fox"], ["the", "dog"]],
                [["Where", "cat"], ["bill", "dogs"]],
                [["bill", "dog"], ["Pen", "dogs"]]
             ]]
        )[0]

        self.assertEqual(features.shape, (4, 6, 6))

        self.assertTrue(np.all(features[0] == 0))
        self.assertEqual(list(features[1, 3, :3]), [1, 0, 0])  # exact match on "dog"
        self.assertEqual(list(features[1, 5, :3]), [1, 0, 0])  # exact batch on "dog"
        self.assertEqual(list(features[2, 2, :3]), [0, 1, 0])  # case match on "bill"
        self.assertEqual(list(features[2, 5, :3]), [0, 0, 1])  # stemmed match on "dogs"

        self.assertEqual(list(features[2, 0, 3:]), [1, 0, 0])  # exact match "Where cat"
        self.assertEqual(list(features[3, 2, 3:]), [0, 1, 0])  # case match "bill dog"
        self.assertEqual(list(features[3, 4, 3:]), [0, 0, 1])  # stemmed match "Pen dogs"
