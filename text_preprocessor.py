import numpy as np
from typing import List, Optional
from collections import Counter

from tqdm import tqdm

from configurable import Configurable
from data_processing.document_splitter import ExtractedParagraphWithAnswers, MergeParagraphs
from data_processing.qa_training_data import ParagraphAndQuestion
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import flatten_iterable


class ParagraphWithAnswers(object):
    def __init__(self, text: List[str], answer_spans: np.ndarray):
        self.text = text
        self.answer_spans = answer_spans


class TextPreprocessor(Configurable):
    """ Preprocess text input, must be deterministic. """

    def encode_paragraphs(self, question: List[str],
                          paragraphs: List[ExtractedParagraphWithAnswers]) -> ParagraphWithAnswers:
        raise NotImplementedError()

    def special_tokens(self) -> List[str]:
        pass


class WithIndicators(TextPreprocessor):
    PARAGRAPH_TOKEN = "%%PARAGRAPH%%"
    DOCUMENT_START_TOKEN = "%%DOCUMENT%%"
    PARAGRAPH_GROUP = "%%PARAGRAPH_GROUP%%"

    def __init__(self, remove_cross_answer: bool=True):
        self.remove_cross_answer = remove_cross_answer

    def special_tokens(self) -> List[str]:
        return [self.PARAGRAPH_TOKEN, self.DOCUMENT_START_TOKEN, self.PARAGRAPH_GROUP]

    def encode_paragraphs(self, question: List[str],
                          paragraphs: List[ExtractedParagraphWithAnswers]) -> ParagraphWithAnswers:
        text = []
        answers = []
        offset = 0
        for paras in paragraphs:
            if paras.start == 0:
                text.append(self.DOCUMENT_START_TOKEN)
            else:
                text.append(self.PARAGRAPH_GROUP)
            offset += 1
            spans = paras.answer_spans + offset

            text += paras.text[0]
            offset += len(paras.text[0])

            for sent in paras.text[1:]:
                if self.remove_cross_answer:
                    remove = np.logical_and(spans[:, 0] < offset, spans[:, 1] >= offset)
                    spans = spans[np.logical_not(remove)]

                spans[spans[:, 0] >= offset, 0] += 1
                spans[spans[:, 1] >= offset, 1] += 1
                text.append(self.PARAGRAPH_TOKEN)
                text += sent
                offset += len(sent) + 1

            answers.append(spans)

        answers = np.concatenate(answers)

        return ParagraphWithAnswers(text, answers)


def check_preprocess():
    data = TriviaQaWebDataset()
    merge = MergeParagraphs(400)
    questions = data.get_dev()
    pre = WithIndicators(False)
    remove_cross = WithIndicators(True)
    rng = np.random.RandomState(0)
    rng.shuffle(questions)

    for q in tqdm(questions[:1000]):
        doc = rng.choice(q.all_docs, 1)[0]
        text = data.evidence.get_document(doc.doc_id, n_tokens=800)
        paras = merge.split_annotated(text, doc.answer_spans)
        built = pre.encode_paragraphs(q.question, paras)

        expected_text = flatten_iterable([flatten_iterable(x.text) for x in paras])
        if expected_text != [x for x in built.text if x not in pre.special_tokens()]:
            raise ValueError()

        expected = flatten_iterable([flatten_iterable(p.text)[s:e+1]
                                     for s, e in p.answer_spans] for p in paras)
        expected = Counter([tuple(x) for x in expected])

        actual = [tuple(built.text[s:e+1]) for s,e in built.answer_spans]
        actual_cleaned = Counter(tuple(z for z in x if z not in pre.special_tokens()) for x in actual)
        if actual_cleaned != expected:
            raise ValueError()

        r_built = remove_cross.encode_paragraphs(q.question, paras)
        rc = Counter(tuple(r_built.text[s:e + 1]) for s, e in r_built.answer_spans)
        removed = Counter()
        for w in actual:
            if all(x not in pre.special_tokens() for x in w):
                removed[w] += 1

        if rc != removed:
            raise ValueError()

if __name__ == "__main__":
    check_preprocess()