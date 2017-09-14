import json
from collections import Counter
from typing import List

import numpy as np
from data_processing.document_splitter import DocumentSplitter, ParagraphFilter, MergeParagraphs, \
    TopTfIdf
from data_processing.preprocessed_corpus import DatasetBuilder, Preprocessor, PreprocessedData
from data_processing.qa_training_data import QaCorpusStats, ParagraphAndQuestionSpec
from data_processing.span_data import TokenSpans
from data_processing.text_utils import NltkPlusStopWords
from dataset import Dataset, ListBatcher, ClusteredBatcher
from trivia_qa.build_span_corpus import TriviaQaSampleWebDataset
from trivia_qa.read_data import TriviaQaQuestion
from trivia_qa.training_data import DocumentParagraphQuestion
from utils import flatten_iterable


class LazyDocumentQuestion(object):
    """ `DocumentQuestion` with paragraphs are not cached in RAM """
    __slots__ = ["question_id", "doc_id", "question", "normalized_aliases", "spans", "doc_ranges"]

    def __init__(self, question_id, doc_id, question: List[str], normalized_aliases, spans, doc_ranges):
        self.question_id = question_id
        self.doc_id = doc_id
        self.question = question
        self.normalized_aliases = normalized_aliases
        self.spans = spans
        self.doc_ranges = doc_ranges

    def n_with_answer(self):
        count = 0
        for s,e in self.doc_ranges:
            count += np.any(np.logical_and(self.spans[:, 0] >= s, self.spans[:, 1] < e))
        return count

    def doc_len(self, ix):
        return self.doc_ranges[ix, 1] - self.doc_ranges[ix, 0]


class QuestionsWithContextInfo(object):
    """ Questions with some cached information about their context paragraphs,
    but without storing the entire context """
    def __init__(self, questions: List[LazyDocumentQuestion], context_counts, max_context_len: int, true_len: int):
        self.questions = questions
        self.context_counts = context_counts
        self.max_context_len = max_context_len
        self.true_len = true_len

    def __add__(self, other):
        out = QuestionsWithContextInfo(self.questions + other.questions,
                                       self.context_counts + other.context_counts,
                                       max(self.max_context_len, self.max_context_len),
                                       self.true_len + other.true_len)
        return out


class LazyQuestionDataset(Dataset):
    def __init__(self, questions: QuestionsWithContextInfo,
                 evidence, splitter,
                 force_answer: float,
                 batch_size: int):
        if len(questions.questions) == 0:
            raise ValueError()
        self.evidence = evidence
        self.splitter = splitter
        self.questions = questions
        self.force_answer = force_answer
        self._batcher = ClusteredBatcher(batch_size, lambda x: x[0].doc_len(x[1]))

    def get_word_counts(self):
        voc = set(self.questions.context_counts)
        for q in self.questions.questions:
            voc.update(q.question)
        return voc

    def get_vocab(self):
        voc = set(self.questions.context_counts)
        for q in self.questions.questions:
            voc.update(q.question)
        return voc

    def get_spec(self):
        max_q_len = max(len(q.question) for q in self.questions.questions)
        return ParagraphAndQuestionSpec(self._batcher.get_fixed_batch_size(), max_q_len,
                                        self.questions.max_context_len, None)

    def get_samples(self, n_examples):
        n_batches = n_examples // self._batcher.batch_size
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        # We first pick a paragraph for each question in the whole so we
        # can cluster by context length accurately
        questions = self.questions.questions
        if self.force_answer == 0:
            out = [(q, np.random.randint(0, len(q.doc_ranges))) for q in questions]
        else:
            out = []
            for q in questions:
                if np.random.random() < self.force_answer:
                    candidates = []
                    for i, (s, e) in enumerate(q.doc_ranges):
                        if np.any(np.logical_and(q.spans[:, 0] >= s, q.spans[:, 1] < e)):
                            candidates.append(i)
                    if len(candidates) == 0:
                        raise ValueError()
                    out.append((q, candidates[np.random.randint(0, len(candidates))]))
                else:
                    out.append((q, np.random.randint(0, len(q.doc_ranges))))

        # Now yield batches while loading the selected paragraph as we go
        for batch in self._batcher.get_epoch(out):
            for i, (q, para_ix) in enumerate(batch):
                start, end = q.doc_ranges[para_ix]
                text = flatten_iterable(self.evidence.get_document(q.doc_id, end))

                word_ix = 0
                if start != 0:
                    for sent_ix, sent in enumerate(text):
                        word_ix += len(sent)
                        if word_ix == start:
                            text = text[sent_ix+1:]
                            break
                        if word_ix > start:
                            raise NotImplementedError()

                if sum(len(s) for s in text) != (end - start):
                    # Sanity check
                    raise RuntimeError(sum(len(s) for s in text), end, start)

                spans = q.spans
                spans = spans[np.logical_and(spans[:, 0] >= start, spans[:, 1] < end)] - start
                batch[i] = DocumentParagraphQuestion(q.question_id, q.doc_id, (start, end), q.question,
                                                     text, TokenSpans(q.normalized_aliases, spans))
            yield batch

    def percent_filtered(self):
        return (self.questions.true_len - len(self.questions.questions)) / self.questions.true_len

    def __len__(self):
        return len(self.questions.questions) // self._batcher.batch_size


class LazyRandomParagraphBuilder(DatasetBuilder, Preprocessor):
    def __init__(self, splitter: DocumentSplitter, para_file: ParagraphFilter,
                 require_an_answer: bool, batch_size: int, force_answer: float = 0):
        self.splitter = splitter
        self.para_filter = para_file
        self.require_an_answer = require_an_answer
        self.force_answer = force_answer
        self.batch_size = batch_size

    def build_dataset(self, data: QuestionsWithContextInfo, corpus, is_train) -> Dataset:
        print("%d/%d (%.4f) were kept" % (len(data.questions), data.true_len, len(data.questions)/data.true_len))
        with_answer = sum(x.n_with_answer() for x in data.questions)
        total = sum(len(x.doc_ranges) for x in data.questions)
        print("%d/%d (%.4f) have answers" % (with_answer, total, with_answer/total))
        return LazyQuestionDataset(data, corpus.evidence, self.splitter, self.force_answer, self.batch_size)

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> object:
        true_len = 0
        splitter = self.splitter
        para_filter = self.para_filter

        pairs = []
        counts = Counter()
        max_context_len = 0
        for q in questions:
            true_len += len(q.all_docs)
            for doc in q.all_docs:
                if len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id, splitter.reads_first_n)
                paras = splitter.split_annotated(text, doc.answer_spans)
                paras = para_filter.prune(q.question, paras)

                if len(paras) == 0:
                    continue
                if self.require_an_answer:
                    if all(len(x.answer_spans) == 0 for x in paras):
                        continue

                max_context_len = max(max_context_len, max(x.n_context_words for x in paras))
                for para in paras:
                    for s in para.text:
                        counts.update(s)
                para_ranges = np.array([(x.start, x.end) for x in paras], dtype=np.int32)
                pairs.append(LazyDocumentQuestion(q.question_id, doc.doc_id, q.question,
                                                  q.answer.all_answers, doc.answer_spans,
                                                  para_ranges))
        return QuestionsWithContextInfo(pairs, counts, max_context_len, true_len)

    def build_stats(self, data: QuestionsWithContextInfo) -> object:
        question_counts = Counter()
        for q in data.questions:
            question_counts.update(q.question)
        return QaCorpusStats(question_counts, data.context_counts, [])


def run_on_sample():
    stop = NltkPlusStopWords()
    data = PreprocessedData(TriviaQaSampleWebDataset(),
                                    LazyRandomParagraphBuilder(
                                 MergeParagraphs(400), TopTfIdf(stop, 4), True, 10, 0.5),
                                    sample=25, sample_dev=20)
    corp = data.get_train_corpus()
    corp.get_word_counts()
    for _,_,x in data.get_train().get_batches(1):
        for q in x:
            if len(q.answer.answer_spans) > 0:
                print(q.question)
                print(q.answer.answer_aliases)
                print([flatten_iterable(q.context)[s:e+1] for s,e in q.answer.answer_spans])
            else:
                print("No Answer!")


if __name__ == "__main__":
    run_on_sample()