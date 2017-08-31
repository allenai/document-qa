import sys
from typing import List, Set, Optional, Tuple, Dict

from data_processing.document_splitter import DocumentSplitter, ParagraphWithAnswers, ParagraphFilter
from data_processing.multi_paragraph_qa import DocumentParagraph, MultiParagraphQuestion
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder, FilteredData
from data_processing.qa_training_data import ParagraphAndQuestion, Answer
from data_processing.span_data import TokenSpans
from trivia_qa.read_data import TriviaQaQuestion
from utils import flatten_iterable, ResourceLoader, split, group


"""
Tools to convert the span corpus into training data we can feed into our model
"""


class DocumentParagraphQuestion(ParagraphAndQuestion):
    def __init__(self, q_id: str, doc_id: str, para_range, question: List[str],
                 context: List[str], answer: Answer):
        super().__init__(context, question, answer, q_id)
        self.doc_id = doc_id
        self.para_range = para_range


class ExtractSingleParagraph(Preprocessor):
    def __init__(self, splitter: DocumentSplitter, para_filter: ParagraphFilter, intern):
        self.splitter = splitter
        self.para_filter = para_filter
        self.intern = intern

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> FilteredData:
        splitter = self.splitter
        paragraph_filter = self.para_filter
        output = []
        read_only = splitter.reads_first_n
        for q in questions:
            for doc in q.all_docs:
                text = evidence.get_document(doc.doc_id, n_tokens=read_only)
                if text is None:
                    raise ValueError(doc.doc_id, evidence.get(doc.doc_id))

                paragraphs = splitter.split_annotated(text, doc.answer_spans)
                if paragraph_filter is not None:
                    paragraphs = paragraph_filter.prune(q.question, paragraphs)
                paragraphs = [x for x in paragraphs if len(x.answer_spans) > 0]
                if len(paragraphs) == 0:
                    continue
                paragraph = paragraphs[0]
                output.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (paragraph.start, paragraph.end),
                                                        q.question, flatten_iterable(paragraph.text),
                                                        TokenSpans(q.answer.all_answers, paragraph.answer_spans)))
        return FilteredData(output, sum(len(x.all_docs) for x in questions))

    def finalize(self, x: FilteredData):
        if self.intern:
            question_map = {}
            for q in x.data:
                q.question_id = sys.intern(q.question_id)
                if q.question_id in question_map:
                    q.question = question_map[q.question_id]
                else:
                    q.question = tuple(sys.intern(w) for w in q.question)
                    question_map[q.question_id] = q.question
                q.doc_id = sys.intern(q.doc_id)
                q.context = [sys.intern(w) for w in q.context]


def intern_mutli_question(questions):
    for q in questions:
        q.question = [sys.intern(x) for x in q.question]
        for para in q.paragraphs:
            para.doc_id = sys.intern(para.doc_id)
            para.text = [sys.intern(x) for x in para.text]


class ExtractMultiParagraphs(Preprocessor):
    def __init__(self, splitter: DocumentSplitter, ranker: ParagraphFilter,
                 intern: bool=False, require_an_answer=True):
        self.intern = intern
        self.splitter = splitter
        self.ranker = ranker
        self.require_an_answer = require_an_answer

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> object:
        true_len = 0
        splitter = self.splitter
        para_filter = self.ranker

        with_paragraphs = []
        for q in questions:
            true_len += len(q.all_docs)
            for doc in q.all_docs:
                if self.require_an_answer and len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id, splitter.reads_first_n)
                paras = splitter.split_annotated(text, doc.answer_spans)
                if para_filter is not None:
                    paras = para_filter.prune(q.question, paras)

                if len(paras) == 0:
                    continue
                if self.require_an_answer:
                    if all(len(x.answer_spans) == 0 for x in paras):
                        continue
                doc_paras = [DocumentParagraph(doc.doc_id, x.start, x.end,
                                               i, x.answer_spans, flatten_iterable(x.text))
                             for i, x in enumerate(paras)]
                with_paragraphs.append(MultiParagraphQuestion(q.question_id, q.question, q.answer.all_answers, doc_paras))

        return FilteredData(with_paragraphs, true_len)

    def finalize(self, q: FilteredData):
        if self.intern:
            intern_mutli_question(q.data)


class ExtractMultiParagraphsPerQuestion(Preprocessor):
    def __init__(self, splitter: DocumentSplitter, ranker: ParagraphFilter,
                 intern: bool=False, require_an_answer=True):
        self.intern = intern
        self.splitter = splitter
        self.ranker = ranker
        self.require_an_answer = require_an_answer

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> object:
        splitter = self.splitter
        para_filter = self.ranker

        with_paragraphs = []
        for q in questions:
            paras = []
            for doc in q.all_docs:
                if self.require_an_answer and len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id, splitter.reads_first_n)
                paras.extend(splitter.split_annotated(text, doc.answer_spans))

            if para_filter is not None:
                paras = para_filter.prune(q.question, paras)

            if len(paras) == 0:
                continue
            if self.require_an_answer:
                if all(len(x.answer_spans) == 0 for x in paras):
                    continue
            doc_paras = [DocumentParagraph("", x.start, x.end,
                                           i, x.answer_spans, flatten_iterable(x.text))
                         for i, x in enumerate(paras)]
            with_paragraphs.append(MultiParagraphQuestion(q.question_id, q.question, q.answer.all_answers, doc_paras))

        return FilteredData(with_paragraphs, len(questions))

    def finalize(self, q: FilteredData):
        if self.intern:
            intern_mutli_question(q.data)

