import string
from typing import List

from data_processing.document_splitter import DocumentSplitter, ParagraphFilter
from data_processing.preprocessed_corpus import Preprocessor
from data_processing.qa_data import Answer
from trivia_qa.read_data import TriviaQaQuestion
from trivia_qa.triviaqa_training_data import ExtractSingleParagraph
from utils import flatten_iterable
import numpy as np


class TextAnswer(Answer):
    def __init__(self, answer_aliases, text_idx):
        self.answer_aliases = answer_aliases
        self.text_idx = text_idx


class TagEntities(ExtractSingleParagraph):
    def __init__(self,
                 bound: int,
                 splitter: DocumentSplitter,
                 para_filter: ParagraphFilter,
                 intern: bool):
        super().__init__(splitter, para_filter, intern)
        self.bound = bound
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])


    def preprocess(self, question: List[TriviaQaQuestion], evidence):
        doc_questions = super().preprocess(question, evidence)

        for q in doc_questions.questions:
            answers = (s, min(s+self.bound, e) for s,e in q.answer.answer_spans)
            span_to_ix = {}
            on_ix = 2  # 0: None, 1: answer
            text = flatten_iterable(q.context)
            text_spans = np.zeros(len(text), self.bound)
            for s in range(len(text)):
                if text[s] in self.skip:
                    continue
                l = min(self.bound, len(text)-s)
                for i in range(l):
                    e = s + i + 1
                    if (s, e - 1) in set(answers):
                        text_spans[s, i] = 1
                        continue
                    span = tuple(text[s:e])
                    ix = span_to_ix.get(span)
                    if ix is not None:
                        text_spans[s, i] = ix
                    else:
                        span_to_ix.get[span] = on_ix
                        text_spans[s, i] = on_ix
                        on_ix += 1
            if not np.any(text_spans == 2):
                raise RuntimeError()
            q.answer = TextAnswer(q.answer.answer_aliases, text_spans)
        return doc_questions
