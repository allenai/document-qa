from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from docqa.data_processing.multi_paragraph_qa import ParagraphWithAnswers, MultiParagraphQuestion, TokenSpanGroup
from docqa.data_processing.preprocessed_corpus import Preprocessor
from docqa.data_processing.qa_training_data import ContextAndQuestion, Answer
from docqa.data_processing.span_data import TokenSpans
from docqa.squad.squad_data import Document
from docqa.text_preprocessor import TextPreprocessor
from docqa.utils import flatten_iterable

"""
Preprocessors for document-level question answering with SQuAD data
"""


class SquadParagraphWithAnswers(ParagraphWithAnswers):

    @classmethod
    def merge(cls, paras: List):
        paras.sort(key=lambda x: x.get_order())
        answer_spans = []
        text = []
        original_text = ""
        spans = []
        for para in paras:
            answer_spans.append(len(text) + para.answer_spans)
            spans.append(len(original_text) + para.spans)
            original_text += para.original_text
            text += para.text

        para = SquadParagraphWithAnswers(text, np.concatenate(answer_spans),
                                         paras[0].doc_id, paras[0].paragraph_num,
                                         original_text, np.concatenate(spans))
        return para

    __slots__ = ["doc_id", "original_text", "paragraph_num", "spans"]

    def __init__(self, text: List[str], answer_spans: np.ndarray, doc_id: str, paragraph_num: int,
                 original_text: str, spans: np.ndarray):
        super().__init__(text, answer_spans)
        self.doc_id = doc_id
        self.original_text = original_text
        self.paragraph_num = paragraph_num
        self.spans = spans

    def get_order(self):
        return self.paragraph_num

    def get_original_text(self, start, end):
        return self.original_text[self.spans[start][0]:self.spans[end][1]]

    def build_qa_pair(self, question, question_id, answer_text, group=None):
        if answer_text is None:
            ans = None
        elif group is None:
            ans = TokenSpans(answer_text, self.answer_spans)
        else:
            ans = TokenSpanGroup(answer_text, self.answer_spans, group)
        # returns a context-and-question equiped with a get_original_text method
        return QuestionAndSquadParagraph(question, ans, question_id, self)


class QuestionAndSquadParagraph(ContextAndQuestion):
    def __init__(self, question: List[str], answer: Optional[Answer], question_id: str, para: SquadParagraphWithAnswers):
        super().__init__(question, answer, question_id, para.doc_id)
        self.para = para

    def get_original_text(self, start, end):
        return self.para.get_original_text(start, end)

    def get_context(self):
        return self.para.text

    @property
    def n_context_words(self) -> int:
        return len(self.para.text)


class SquadTfIdfRanker(Preprocessor):
    """
    TF-IDF ranking for SQuAD, this does the same thing as `TopTfIdf`, but its supports efficient usage
    when have many many questions per document
    """

    def __init__(self, stop, n_to_select: int, force_answer: bool, text_process: TextPreprocessor=None):
        self.stop = stop
        self.n_to_select = n_to_select
        self.force_answer = force_answer
        self.text_process = text_process
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)

    def preprocess(self, question: List[Document], evidence):
        return self.ranked_questions(question)

    def rank(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        tfidf = self._tfidf
        para_features = tfidf.fit_transform([" ".join(" ".join(s) for s in x) for x in paragraphs])
        q_features = tfidf.transform([" ".join(q) for q in questions])
        scores = pairwise_distances(q_features, para_features, "cosine")
        return scores

    def ranked_questions(self, docs: List[Document]) -> List[MultiParagraphQuestion]:
        out = []
        for doc in docs:
            scores = self.rank(flatten_iterable([q.words for q in x.questions] for x in doc.paragraphs),
                               [x.text for x in doc.paragraphs])
            q_ix = 0
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    para_scores = scores[q_ix]
                    para_ranks = np.argsort(para_scores)
                    selection = [i for i in para_ranks[:self.n_to_select]]

                    if self.force_answer and para_ix not in selection:
                        selection[-1] = para_ix

                    para = []
                    for ix in selection:
                        if ix == para_ix:
                            ans = q.answer.answer_spans
                        else:
                            ans = np.zeros((0, 2), dtype=np.int32)
                        p = doc.paragraphs[ix]
                        if self.text_process:
                            text, ans, inv = self.text_process.encode_paragraph(q.words,  [flatten_iterable(p.text)],
                                                               p.paragraph_num == 0, ans, p.spans)
                            para.append(SquadParagraphWithAnswers(text, ans, doc.doc_id,
                                                                  ix, p.original_text, inv))
                        else:
                            para.append(SquadParagraphWithAnswers(flatten_iterable(p.text), ans, doc.doc_id,
                                                                  ix, p.original_text, p.spans))

                    out.append(MultiParagraphQuestion(q.question_id, q.words, q.answer.answer_text, para))
                    q_ix += 1
        return out
