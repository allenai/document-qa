"""
Data for cases where we have many paragraphs mapped to a single question
"""
from typing import List
from collections import Counter

from data_processing.preprocessed_corpus import DatasetBuilder, FilteredData
from data_processing.qa_training_data import ParagraphAndQuestionSpec, WordCounts, ParagraphAndQuestion
from data_processing.span_data import TokenSpans
from dataset import Dataset, ListBatcher
import numpy as np


class DocumentParagraph(object):
    __slots__ = ["doc_id", "start", "end", "answer_spans", "text", "rank"]

    def __init__(self, doc_id: str, start: int, end: int, rank: int,
                 answer_spans: np.ndarray, text: List[str]):
        self.doc_id = doc_id
        self.start = start
        self.rank = rank
        self.end = end
        self.answer_spans = answer_spans
        self.text = text


class MultiParagraphQuestion(object):
    __slots__ = ["question_id", "question", "end", "answer_text", "paragraphs"]

    def __init__(self, question_id: str, question: List[str], answer_text: List[str],
                 paragraphs: List[DocumentParagraph]):
        self.question_id = question_id
        self.question = question
        self.answer_text = answer_text
        self.paragraphs = paragraphs


class RandomParagraphDataset(Dataset):
    def __init__(self,
                 questions: List[MultiParagraphQuestion],
                 force_answer: float,
                 true_len: int,
                 batcher: ListBatcher):
        self.questions = questions
        self.force_answer = force_answer
        self.batcher = batcher
        self.true_len = true_len

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in q.paragraphs:
                voc.update(para.text)
        return voc

    def get_spec(self):
        max_q_len = max(len(q.question) for q in self.questions)
        max_c_len = max(max(len(p.text) for p in q.paragraphs) for q in self.questions)
        return ParagraphAndQuestionSpec(self.batcher.get_fixed_batch_size(), max_q_len,
                                        max_c_len, None)

    def get_samples(self, n_examples):
        n_batches = n_examples // self.batcher.batch_size
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        # We first pick a paragraph for each question in the entire training set so we
        # can cluster by context length accurately
        questions = self.questions
        out = []
        for q in questions:
            if self.force_answer > 0 and np.random.random() < self.force_answer:
                candidates = [x for x in q.paragraphs if len(x.answer_spans) > 0]
                if len(candidates) == 0:
                    raise ValueError()
            else:
                candidates = q.paragraphs
            selected = candidates[np.random.randint(0, len(candidates))]
            out.append(ParagraphAndQuestion(selected.text, q.question,
                                            TokenSpans(q.answer_text, selected.answer_spans), q.question_id))

        return self.batcher.get_epoch(out)

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))


class RandomParagraphDatasetBuilder(DatasetBuilder):
    """ For use with the preprocessing module """
    def __init__(self, train_batcher: ListBatcher, eval_batcher: ListBatcher,
                 force_answer: float):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher
        self.force_answer = force_answer

    def build_stats(self, data: FilteredData):
        wc = Counter()
        for point in data.data:
            wc.update(point.question)
            for para in point.paragraphs:
                for sent in para.text:
                    wc.update(sent)
        return WordCounts(wc)

    def build_dataset(self, data: FilteredData, corpus, is_train: bool) -> Dataset:
        return RandomParagraphDataset(data.data, self.force_answer, data.true_len,
                                      self.train_batcher if is_train else self.eval_batcher)
