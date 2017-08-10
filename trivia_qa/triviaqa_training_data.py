import json
import sys
from collections import Counter
from typing import List, Set, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

from data_processing.document_splitter import DocumentSplitter, ExtractedParagraph, ParagraphFilter
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder
from data_processing.qa_data import ParagraphAndQuestionSpec, ParagraphAndQuestionDataset, Answer, Batcher
from dataset import Dataset, ListBatcher
from trivia_qa.build_span_corpus import TriviaQaWebDataset, TriviaQaSpanCorpus
from trivia_qa.read_data import TriviaQaQuestion
from utils import flatten_iterable, ResourceLoader, split, partition


class TriviaQaAnswer(Answer):
    __slots__ = ["answer_spans", "answer_aliases"]

    def __init__(self, answer_spans, answer_aliases):
        self.answer_spans = answer_spans
        self.answer_aliases = answer_aliases

    def get_vocab(self):
        return []


class DocumentParagraphQuestion(object):
    __slots__ = ["q_id", "doc_id", "para_range", "question", "context", "answer"]

    def __init__(self, q_id: str, doc_id: str, para_range, question: List[str],
                 context: List[List[str]], answer: TriviaQaAnswer):
        self.q_id = q_id
        self.doc_id = doc_id
        self.para_range = para_range
        self.question = question
        self.context = context
        self.answer = answer

    @property
    def question_id(self):
        return self.q_id, self.doc_id


class DocumentQuestion(object):
    """ Question that might have many paragraphs with answers in its document """
    __slots__ = ["q_id", "doc_id", "question", "normalized_aliases", "paragraphs"]

    def __init__(self, q_id, doc_id, question: List[str], normalized_aliases, paragraphs: List[ExtractedParagraph]):
        if len(paragraphs) == 0 or len(question) == 0:
            raise ValueError()
        self.q_id = q_id
        self.doc_id = doc_id
        self.question = question
        self.normalized_aliases = normalized_aliases
        self.paragraphs = paragraphs

    def all_paragraphs(self) -> List[DocumentParagraphQuestion]:
        output = []
        for paragraph in self.paragraphs:
            ans = TriviaQaAnswer(paragraph.answer_spans, self.normalized_aliases)
            output.append(DocumentParagraphQuestion(self.q_id, self.doc_id,
                                      (paragraph.start, paragraph.end), self.question,
                                      paragraph.text, ans))
        return output

    def random_paragraph(self) -> DocumentParagraphQuestion:
        paragraph = self.paragraphs[np.random.randint(len(self.paragraphs))]
        ans = TriviaQaAnswer(paragraph.answer_spans, self.normalized_aliases)
        return DocumentParagraphQuestion(self.q_id, self.doc_id,
                                         (paragraph.start, paragraph.end), self.question,
                                         paragraph.text, ans)


class PrunedQuestions(object):
    def __init__(self, questions: List[DocumentParagraphQuestion], true_len: int):
        self.true_len = true_len
        self.questions = questions

    def __add__(self, other):
        return PrunedQuestions(other.questions + self.questions, other.true_len + self.true_len)


class DocCorpusLazyStats(object):
    def __init__(self, data: List[DocumentQuestion], special_tokens=None):
        self.data = data
        self.special_tokens = special_tokens
        self._question_counts = None
        self._context_counts = None

    def get_question_counts(self):
        if self._question_counts is None:
            counter = Counter()
            for q in self.data:
                counter.update(q.question)
            self._question_counts = counter
        return self._question_counts

    def get_context_counts(self):
        if self._context_counts is None:
            counter = Counter()
            for q in self.data:
                for para in q.paragraphs:
                    counter.update(flatten_iterable(para.text))
            self._context_counts = counter
        return self._context_counts


class DocParaCorpusLazyStats(object):
    def __init__(self, data: List[DocumentParagraphQuestion], special_tokens=None):
        self.data = data
        self.special_tokens = special_tokens
        self._question_counts = None
        self._context_counts = None

    def get_question_counts(self):
        if self._question_counts is None:
            counter = Counter()
            for q in self.data:
                counter.update(q.question)
            self._question_counts = counter
        return self._question_counts

    def get_context_counts(self):
        if self._context_counts is None:
            counter = Counter()
            for q in self.data:
                counter.update(flatten_iterable(q.context))
            self._context_counts = counter
        return self._context_counts

    def get_word_counts(self):
        return self.get_question_counts() + self.get_context_counts()


class RandomParagraphDataset(Dataset):
    def __init__(self, questions: List[DocumentQuestion], n_total, batcher: ListBatcher):
        self.questions = questions
        self.n_total = n_total
        self.batcher = batcher

    def get_spec(self) -> ParagraphAndQuestionSpec:
        max_question_words = max(len(x.question) for x in self.questions)
        max_context_words = max(max(para.n_context_words for para in q.paragraphs) for q in self.questions)
        max_word_size = max(max(len(w) for w in q.question) for q in self.questions)
        for q in self.questions:
            for para in q.paragraphs:
                max_word_size = max(max_word_size,
                                max(max(len(w) for w in sent) for sent in para.text if len(sent) > 0))
        return ParagraphAndQuestionSpec(self.batcher.get_fixed_batch_size, max_question_words,
                                        max_context_words, None, None,
                                        max_word_size)

    def get_vocab(self) -> Set[str]:
        voc = set()
        for point in self.questions:
            voc.update(point.question)
            for para in point.paragraphs:
                for sent in para.text:
                    voc.update(sent)
        return voc

    def _gen_epoch(self):
        return [x.random_paragraph() for x in self.questions]

    def percent_filtered(self):
        return (self.n_total - len(self.questions)) / self.n_total

    def get_epoch(self):
        return self.batcher.get_epoch([x.random_paragraph() for x in self.questions])

    def __len__(self):
        return len(self.questions)


def intern_questions(questions: List[DocumentParagraphQuestion]):
    question_map = {}
    for q in questions:
        q.q_id = sys.intern(q.q_id)
        if q.q_id in question_map:
            q.question = question_map[q.q_id]
        else:
            q.question = tuple(sys.intern(w) for w in q.question)
            question_map[q.q_id] = q.question
        q.doc_id = sys.intern(q.doc_id)
        for ix, sent in enumerate(q.context):
            q.context[ix] = [sys.intern(w) for w in sent]


class ExtractSingleParagraph(Preprocessor):
    def __init__(self, splitter: DocumentSplitter, para_filter: ParagraphFilter, intern):
        self.splitter = splitter
        self.para_filter = para_filter
        self.intern = intern

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> object:
        splitter = self.splitter
        paragraph_filter = self.para_filter
        output = []
        read_only = splitter.reads_first_n
        for q in questions:
            for doc in q.all_docs:
                text = evidence.get_document(doc.doc_id, n_tokens=read_only)
                if text is None:
                    raise ValueError(doc.doc_id, evidence.get(doc.doc_id))

                paragraphs = splitter.split(text, doc.answer_spans)
                if paragraph_filter is not None:
                    paragraphs = paragraph_filter.prune(q.question, paragraphs)
                paragraphs = [x for x in paragraphs if len(x.answer_spans) > 0]
                if len(paragraphs) == 0:
                    continue
                paragraph = paragraphs[0]
                output.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (paragraph.start, paragraph.end),
                                                        q.question, paragraph.text,
                                               TriviaQaAnswer(paragraph.answer_spans, q.answer.all_answers)))
        return PrunedQuestions(output, sum(len(x.all_docs) for x in questions))

    def finalize(self, x: PrunedQuestions):
        if self.intern:
            intern_questions(x.questions)


class ExtractPrecomputedParagraph(Preprocessor):
    def __init__(self, source_files, first_only: bool, intern: bool=True):
        self.source_files = source_files
        self.first_only = first_only
        self.intern = intern
        self._cached_order = None

    def get_cache(self):
        if self._cached_order is None:
            self._cached_order = {}
            for filename in self.source_files:
                with open(filename, "r") as f:
                    data = json.load(f)
                for point in data:
                    self._cached_order[(point["quid"], point["doc_id"])] = np.array(point["spans"], dtype=np.int32)
        return self._cached_order

    def set_cache(self, c):
        self._cached_order = c

    def preprocess(self, questions: List[TriviaQaQuestion], evidence):
        paragraphs = self.get_cache()
        out = []
        for q in questions:
            for doc in q.all_docs:
                ans = doc.answer_spans
                if len(ans) == 0:
                    continue
                spans = paragraphs.get((q.question_id, doc.doc_id))
                if spans is None:
                    continue
                if self.first_only:
                    spans = spans[:1]
                start, end, answer_spans = None, None, None
                for s,e in spans:
                    answer_spans = ans[np.logical_and(ans[:, 0] >= s, ans[:, 1] < e)]
                    if len(answer_spans) > 0:
                        start, end = s, e
                        break
                if start is None:
                    continue
                text = evidence.get_document(doc.doc_id, n_tokens=end, flat=True)[start:]
                if len(text) != end-start:
                    raise ValueError()
                out.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (start, end), q.question, text,
                                               TriviaQaAnswer(answer_spans, q.answer.all_answers)))
        return PrunedQuestions(out, sum(len(x.all_docs) for x in questions))

    def finalize(self, x: PrunedQuestions):
        if self.intern:
            intern_questions(x.questions)


class InMemoryWebQuestionBuilder(DatasetBuilder):
    def __init__(self, train_batcher: ListBatcher, eval_batcher: ListBatcher):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher

    def build_stats(self, data: PrunedQuestions) -> object:
        return DocParaCorpusLazyStats(data.questions, [])

    def build_dataset(self, data: PrunedQuestions, evidence, is_train: bool) -> Dataset:
        points, true_len = data.questions, data.true_len
        print("Kept %d/%d (%.4f) doc/q pairs" % (len(points), true_len, len(points)/true_len))
        return ParagraphAndQuestionDataset(data.questions, self.train_batcher if is_train else self.eval_batcher, data.true_len)

