import unittest
from typing import List
import numpy as np

from docqa.data_processing.document_splitter import DocumentSplitter, ExtractedParagraph, extract_tokens
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.utils import flatten_iterable


class RandomSplitter(DocumentSplitter):
    def split(self, doc: List[List[List[str]]]) -> List[ExtractedParagraph]:
        words = flatten_iterable(flatten_iterable(doc))
        on_word = 0
        out = []
        while True:
            end_word = on_word + np.random.randint(1, 7)
            if on_word + end_word > len(words):
                out.append(ExtractedParagraph([words[on_word:]], on_word, len(words)))
                return out
            out.append(ExtractedParagraph([words[on_word:end_word]], on_word, end_word))
            on_word = end_word


class TestSplitter(unittest.TestCase):

    def test_split_inv(self):
        paras = [
            "One fish two fish. Red fish blue fish",
            "Just one sentence",
            "How will an overhead score? The satisfactory juice returns against an inviting protein. "
            "How can a rat expand? The subway fishes throughout a struggle. The guaranteed herd pictures an "
            "episode into the accustomed damned. The garbage reigns beside the component!",
        ]
        tok = NltkAndPunctTokenizer()
        tokenized = [tok.tokenize_with_inverse(x) for x in paras]
        inv_split = RandomSplitter().split_inverse(tokenized)
        for para in inv_split:
            self.assertTrue(flatten_iterable(para.text) == [para.original_text[s:e] for s,e in para.spans])

