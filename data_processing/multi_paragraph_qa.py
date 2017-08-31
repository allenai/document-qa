"""
Data for cases where we have many paragraphs mapped to a single question
"""
from typing import List
from collections import Counter, defaultdict

from data_processing.preprocessed_corpus import DatasetBuilder, FilteredData
from data_processing.qa_training_data import ParagraphAndQuestionSpec, WordCounts, ParagraphAndQuestion, \
    ContextAndQuestion, ParagraphAndQuestionDataset
from data_processing.span_data import TokenSpans
from dataset import Dataset, ListBatcher, ClusteredBatcher
import numpy as np

from utils import flatten_iterable


class QuestionAndContexts(object):
    """ Generic training point for questions with multiple paragraphs as context """
    def __init__(self, question: List[str], quid: str, answer_text: List[str], answer_spans: List[np.ndarray],
                 contexts: List[List[List[str]]]):
        self.quid = quid
        self.question = question
        self.answer_text = answer_text
        self.answer_spans = answer_spans
        self.contexts = contexts


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


class SampleQuestionDataset(ParagraphAndQuestionDataset):
    def __init__(self, data: List[MultiParagraphQuestion],
                 batching: ListBatcher, sample_doc=True):
        self.grouped = defaultdict(list)
        flattened = []
        for point in data:
            for para in point.paragraphs:
                pair = ParagraphAndQuestion(para.text, point.question,
                                             TokenSpans(point.answer_text, para.answer_spans),
                                             point.question_id, para.doc_id)
                if sample_doc:
                    self.grouped[(point.question_id, para.doc_id)].append(pair)
                else:
                    self.grouped[point.question_id].append(pair)
                flattened.append(pair)
        print("Built %d groups" % len(self.grouped))
        super().__init__(flattened, batching)

    def get_samples(self, n_examples):
        print("Sampled %d questions, to a total of %d elements" % (len(self.grouped), n_examples))
        all_ids = list(self.grouped)
        # we can't sample directly since numpy will interpret the list of tuple as a 2d array
        ids = [all_ids[i] for i in np.random.choice(len(all_ids), n_examples, replace=False)]
        out = flatten_iterable(self.grouped[k] for k in ids)

        return self.batching.get_epoch(out), self.batching.epoch_size(len(out))


class TokenSpanGroup(TokenSpans):
    def __init__(self, answer_text: List[str], answer_spans: np.ndarray, group_id: int):
        super().__init__(answer_text, answer_spans)
        self.group_id = group_id


class ParagraphSelection(object):
    def __init__(self, question: MultiParagraphQuestion, selection):
        self.question = question
        self.selection = selection
        self.n_context_words = max(len(question.paragraphs[i].text) for i in selection)


class RandomParagraphSetDataset(Dataset):
    """
    Sample multiple paragraphs for each question and include them in the same batch
    """

    def __init__(self,
                 questions: List[MultiParagraphQuestion],
                 n_paragraphs,
                 force_answer: bool,
                 true_len: int,
                 batch_size: int):
        self.n_paragraphs = n_paragraphs
        self.questions = questions
        self.force_answer = force_answer
        self.true_len = true_len
        self.batcher = ClusteredBatcher(batch_size, lambda x: x.n_context_words, truncate_batches=True)

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in q.paragraphs:
                voc.update(para.text)
        return voc

    def get_spec(self):
        max_q_len = max(len(q.question) for q in self.questions)
        max_c_len = max(self.n_paragraphs*max(len(p.text) for p in q.paragraphs) for q in self.questions)
        return ParagraphAndQuestionSpec(None, max_q_len, max_c_len, None)

    def get_epoch(self):
        return self._build_expanded_batches(self.questions)

    def _build_expanded_batches(self, questions):
        # We first pick paragraph(s) for each question in the entire training set so we
        # can cluster by context length accurately
        out = []
        for q in questions:
            if len(q.paragraphs) == 1:
                selected = [0]
            elif self.force_answer:
                with_answer = [i for i, p in enumerate(q.paragraphs) if len(p.answer_spans) > 0]
                answer_selection = with_answer[np.random.randint(len(with_answer))]
                other = np.array([i for i, x in enumerate(q.paragraphs) if i != answer_selection])
                selected = np.random.choice(other, min(len(other), self.n_paragraphs-1), replace=False)
                selected = np.insert(selected, 0, answer_selection)
            else:
                selected = np.random.choice(len(q.paragraphs), min(len(q.paragraphs), self.n_paragraphs))

            out.append(ParagraphSelection(q, selected))

        out.sort(key=lambda x: x.n_context_words)

        group = 0
        for selection_batch in self.batcher.get_epoch(out):
            batch = []
            for selected in selection_batch:
                q = selected.question
                for i in selected.selection:
                    para = q.paragraphs[i]
                    batch.append(ParagraphAndQuestion(para.text, q.question,
                                            TokenSpanGroup(q.answer_text, para.answer_spans, group),
                                            q.question_id, para.doc_id))
                group += 1
            yield batch

    def get_samples(self, n_examples):
        n_batches = self.batcher.epoch_size(n_examples)
        return self._build_expanded_batches(np.random.choice(self.questions, n_examples, replace=False)), n_batches

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))


def multi_paragraph_word_counts(data, special_tokens=None):
    wc = Counter()
    for point in data:
        wc.update(point.question)
        for para in point.paragraphs:
            for sent in para.text:
                wc.update(sent)
    return WordCounts(wc, special_tokens)


class IndividualParagraphBuilder(DatasetBuilder):
    """ Treat each paragraph as its own training point """

    def __init__(self, train_batcher: ListBatcher, eval_batcher: ListBatcher,
                 force_answer: float):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher
        self.force_answer = force_answer

    def build_stats(self, data: FilteredData):
        return multi_paragraph_word_counts(data.data)

    def build_dataset(self, data: FilteredData, corpus, is_train: bool) -> Dataset:
        flattened = []
        for point in data.data:
            for para in point.paragraphs:
                flattened.append(ParagraphAndQuestion(para.text, point.question,
                                            TokenSpans(point.answer_text, para.answer_spans),
                                            point.question_id))
        return ParagraphAndQuestionDataset(flattened, self.train_batcher if is_train else self.eval_batcher)


class RandomParagraphDatasetBuilder(DatasetBuilder):
    """
    Select a random paragraph each iteration. The advantage of doing this over flattening the dataset
    is that we do not biassing the training towards questions with more more paragraph, and we can
    sample in favour of paragraphs that contain and an answer
    """

    def __init__(self, train_batcher: ListBatcher, eval_batcher: ListBatcher,
                 force_answer: float, sample_questions: bool=False):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher
        self.force_answer = force_answer
        self.sample_eval_questions = sample_questions

    def build_stats(self, data: FilteredData):
        return multi_paragraph_word_counts(data.data)

    def build_dataset(self, data: FilteredData, corpus, is_train: bool) -> Dataset:
        if is_train:
            return RandomParagraphDataset(data.data, self.force_answer, data.true_len, self.train_batcher)
        elif self.sample_eval_questions:
            return SampleQuestionDataset(data.data, self.eval_batcher, True)
        else:
            return RandomParagraphDataset(data.data, self.force_answer, data.true_len, self.eval_batcher)


class RandomParagraphSetDatasetBuilder(DatasetBuilder):
    """
    Select several random paragraph(s) each iteration, and include them the same batch
    Also labels paragraphs in the same batch with the `group` they are in
    """

    def __init__(self, train_batch_size: int, test_batch_size: int, n_paragraphs: int, force_answer: bool):
        self.n_paragraphs = n_paragraphs
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.force_answer = force_answer

    def build_stats(self, data: FilteredData):
        return multi_paragraph_word_counts(data.data)

    def build_dataset(self, data: FilteredData, corpus, is_train: bool) -> Dataset:
        return RandomParagraphSetDataset(data.data, self.n_paragraphs, self.force_answer,
                                         data.true_len,
                                         self.train_batch_size if is_train else self.test_batch_size)


class ConcatParagraphDatasetBuilder(DatasetBuilder):
    """ Concat all paragraphs in to one "mega" paragraph, possibly with a deliminator """

    PARAGRAPH_TOKEN = "%%PARAGRAPH%%"

    def __init__(self, train_batcher: ListBatcher, eval_batcher: ListBatcher,
                 paragraph_start_tokens: bool):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher
        self.paragraph_start_tokens = paragraph_start_tokens

    def build_stats(self, data: FilteredData):
        tokens = [self.PARAGRAPH_TOKEN] if self.paragraph_start_tokens else None
        return multi_paragraph_word_counts(data.data, tokens)

    def build_dataset(self, data: FilteredData, corpus, is_train: bool) -> Dataset:
        concatenated = []
        for point in data.data:
            text = []
            answer_spans = []
            for paragraph in point.paragraphs:
                if self.paragraph_start_tokens:
                    text.append(self.PARAGRAPH_TOKEN)
                answer_spans.append(paragraph.answer_spans + len(text))
                text += paragraph.text
            doc_id = point.paragraphs[0].doc_id
            concatenated.append(ParagraphAndQuestion(text, point.question,
                                            TokenSpans(point.answer_text, np.concatenate(answer_spans)),
                                            point.question_id, doc_id))
        return ParagraphAndQuestionDataset(concatenated,
                                           self.train_batcher if is_train else self.eval_batcher,
                                           data.true_len)
