from collections import Counter
from typing import List, Union

import numpy as np
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, WordCounts, ParagraphAndQuestion, \
    ContextAndQuestion, ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.dataset import Dataset, ListBatcher, ClusteredBatcher

from docqa.data_processing.preprocessed_corpus import DatasetBuilder, FilteredData

"""
Data for cases where we have many paragraphs mapped to a single question
"""


class ParagraphWithAnswers(object):
    __slots__ = ["text", "answer_spans"]

    def __init__(self, text: List[str], answer_spans: np.ndarray):
        self.text = text
        self.answer_spans = answer_spans

    @classmethod
    def merge(cls, paras: List):
        paras.sort(key=lambda x: x.get_order())
        answer_spans = []
        text = []
        for para in paras:
            answer_spans.append(len(text) + para.answer_spans)
            text += para.text
        return ParagraphWithAnswers(text, np.concatenate(answer_spans))

    def get_context(self):
        return self.text

    def get_order(self):
        raise NotImplementedError()

    def build_qa_pair(self, question, question_id, answer_text, group=None) -> ContextAndQuestion:
        if answer_text is None:
            ans = None
        elif group is None:
            ans = TokenSpans(answer_text, self.answer_spans)
        else:
            ans = TokenSpanGroup(answer_text, self.answer_spans, group)
        return ParagraphAndQuestion(self.text, question, ans, question_id)


class DocumentParagraph(ParagraphWithAnswers):
    __slots__ = ["doc_id", "start", "end", "rank"]

    def __init__(self, doc_id: str, start: int, end: int, rank: int,
                 answer_spans: np.ndarray, text: List[str]):
        super().__init__(text, answer_spans)
        self.doc_id = doc_id
        self.start = start
        self.rank = rank
        self.end = end

    def get_order(self):
        return self.start


class MultiParagraphQuestion(object):
    """ Question associated with multiple paragraph w/pre-computed answer spans """

    __slots__ = ["question_id", "question", "end", "answer_text", "paragraphs"]

    def __init__(self, question_id: str, question: List[str], answer_text: List[str],
                 paragraphs: List[ParagraphWithAnswers]):
        self.question_id = question_id
        self.question = question
        self.answer_text = answer_text
        self.paragraphs = paragraphs


class RandomParagraphDataset(Dataset):
    """ Samples a random set of paragraphs from question to build question-paragraph pairs """

    def __init__(self,
                 questions: List[MultiParagraphQuestion],
                 force_answer: float,
                 true_len: int,
                 n_to_sample: int,
                 batcher: ListBatcher):
        self.questions = questions
        self.n_to_sample = n_to_sample
        self.force_answer = force_answer
        self.batcher = batcher
        self.true_len = true_len
        self._n_examples = sum(min(self.n_to_sample, len(q.paragraphs)) for q in questions)

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
        return ParagraphAndQuestionSpec(None if self.n_to_sample != 1 else self.batcher.get_fixed_batch_size(),
                                        max_q_len, max_c_len, None)

    def get_samples(self, n_examples):
        n_batches = n_examples // self.batcher.get_max_batch_size()
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        # We first pick a paragraph for each question in the entire training set so we
        # can cluster by context length accurately
        questions = self.questions
        out = []
        for q in questions:
            if len(q.paragraphs) <= self.n_to_sample:
                selected = q.paragraphs
            elif self.force_answer == 0:
                selected = np.random.choice(q.paragraphs, self.n_to_sample, replace=False)
            else:
                answer_probs = np.array([len(p.answer_spans) > 0 for p in q.paragraphs], dtype=np.float64)
                answer_probs /= answer_probs.sum()
                uniform_probs = np.full(len(q.paragraphs), 1.0/len(q.paragraphs))
                probs = (answer_probs + uniform_probs) / 2.0
                selected = np.random.choice(q.paragraphs, self.n_to_sample, p=probs, replace=False)
            for s in selected:
                out.append(s.build_qa_pair(q.question, q.question_id, q.answer_text))

        return self.batcher.get_epoch(out)

    def percent_filtered(self):
        return 0

    def __len__(self):
        return self.batcher.epoch_size(self._n_examples)


class StratifyParagraphsDataset(Dataset):
    """
    Samples paragraph for each question to build question-paragraph pairs, but
    stratify the sampling across epochs so paragraphs are seen at about the same rate
    """

    def __init__(self,
                 questions: List[MultiParagraphQuestion],
                 true_len: int,
                 overample_first_answer: List[int],
                 batcher: ListBatcher):
        self.questions = questions
        self.overample_first_answer = overample_first_answer
        self.batcher = batcher
        self.true_len = true_len

        self._order = []
        self._on = np.zeros(len(questions), dtype=np.int32)
        for i in range(len(questions)):
            paras = questions[i].paragraphs
            order = list(range(len(paras)))
            if len(self.overample_first_answer) > 0:
                ix = 0
                for i, p in enumerate(paras):
                    if len(p.answer_spans) > 0:
                        order += [i] * self.overample_first_answer[ix]
                        ix += 1
                        if ix >= len(self.overample_first_answer):
                            break

            order = np.array(order, dtype=np.int32)
            np.random.shuffle(order)
            self._order.append(order)

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
        n_batches = n_examples // self.batcher.get_max_batch_size()
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        questions = self.questions
        out = []
        for i, q in enumerate(questions):
            order = self._order[i]
            selected = q.paragraphs[order[self._on[i]]]
            self._on[i] += 1
            if self._on[i] == len(order):
                np.random.shuffle(order)
                self._on[i] = 0

            out.append(selected.build_qa_pair(q.question, q.question_id, q.answer_text))

        return self.batcher.get_epoch(out)

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))

    def __setstate__(self, state):
        if "oversample_answer" in state:
            raise ValueError()
        super().__setstate__(state)


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
                 questions: List[MultiParagraphQuestion], true_len: int, n_paragraphs: int,
                 batch_size: int, mode: str, force_answer: bool,
                 oversample_first_answer: List[int]):
        self.mode = mode
        self.questions = questions
        self.force_answer = force_answer
        self.true_len = true_len
        self.n_paragraphs = n_paragraphs
        self.oversample_first_answer = oversample_first_answer
        self._n_pairs = sum(min(len(q.paragraphs), n_paragraphs) for q in questions)
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
        max_c_len = max(max(len(p.text) for p in q.paragraphs) for q in self.questions)
        return ParagraphAndQuestionSpec(self.batcher.get_fixed_batch_size() if self.mode == "merge" else None,
                                        max_q_len, max_c_len, None)

    def get_epoch(self):
        return self._build_expanded_batches(self.questions)

    def _build_expanded_batches(self, questions):
        # We first pick paragraph(s) for each question in the entire training set so we
        # can cluster by context length accurately
        out = []
        for q in questions:
            if len(q.paragraphs) <= self.n_paragraphs:
                selected = np.arange(len(q.paragraphs))
            elif not self.force_answer and len(self.oversample_first_answer) == 0:
                selected = np.random.choice(len(q.paragraphs), self.n_paragraphs, replace=False)
            else:
                if not self.force_answer:
                    raise NotImplementedError()
                with_answer = [i for i, p in enumerate(q.paragraphs) if len(p.answer_spans) > 0]
                for ix, over_sample in zip(list(with_answer), self.oversample_first_answer):
                    with_answer += [ix] * over_sample
                answer_selection = with_answer[np.random.randint(len(with_answer))]
                other = np.array([i for i, x in enumerate(q.paragraphs) if i != answer_selection])
                selected = np.random.choice(other, min(len(other), self.n_paragraphs-1), replace=False)
                selected = np.insert(selected, 0, answer_selection)

            if self.mode == "flatten":
                for i in selected:
                    out.append(q.paragraphs[i].build_qa_pair(q.question, q.question_id, q.answer_text))
            else:
                out.append(ParagraphSelection(q, selected))

        out.sort(key=lambda x: x.n_context_words)

        if self.mode == "flatten":
            for batch in self.batcher.get_epoch(out):
                yield batch
        elif self.mode == "group":
            group = 0
            for selection_batch in self.batcher.get_epoch(out):
                batch = []
                for selected in selection_batch:
                    q = selected.question
                    for i in selected.selection:
                        para = q.paragraphs[i]
                        batch.append(para.build_qa_pair(q.question, q.question_id, q.answer_text, group))
                    group += 1
                yield batch
        elif self.mode == "merge":
            for selection_batch in self.batcher.get_epoch(out):
                batch = []
                for selected in selection_batch:
                    q = selected.question
                    paras = [q.paragraphs[i] for i in selected.selection]
                    para = paras[0].merge(paras)
                    batch.append(para.build_qa_pair(q.question, q.question_id, q.answer_text))
                yield batch
        else:
            raise RuntimeError()

    def get_samples(self, n_examples):
        questions = np.random.choice(self.questions, n_examples, replace=False)
        if self.mode == "flatten":
            n_batches = self.batcher.epoch_size(sum(min(len(q.paragraphs), self.n_paragraphs) for q in questions))
        else:
            n_batches = self.batcher.epoch_size(n_examples)
        return self._build_expanded_batches(np.random.choice(questions, n_examples, replace=False)), n_batches

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        if self.mode == "flatten":
            return self.batcher.epoch_size(self._n_pairs)
        else:
            return self.batcher.epoch_size(len(self.questions))


class StratifiedParagraphSetDataset(Dataset):
    """
    Sample multiple paragraphs each epoch and include them in the same batch,
    but stratify the sampling across epochs
    """

    def __init__(self,
                 questions: List[MultiParagraphQuestion],
                 true_len: int,
                 batch_size: int,
                 force_answer: bool,
                 oversample_first_answer: List[int],
                 merge: bool):
        """
        :param true_len: Number questions before any filtering was done
        :param batch_size: Batch size to use
        :param force_answer: Require an answer exists for at least
        one paragraph for each question each batch
        :param oversample_first_answer: Over sample the top-ranked answer-containing paragraphs
        by duplicating them the specified amount
        :param merge: Merge all selected paragraphs for each question into a single super-paragraph
        """
        self.overample_first_answer = oversample_first_answer
        self.questions = questions
        self.merge = merge
        self.true_len = true_len
        self.batcher = ClusteredBatcher(batch_size, lambda x: x.n_context_words, truncate_batches=True)
        self._order = []
        self._on = np.zeros(len(questions), dtype=np.int32)
        for q in questions:
            if len(q.paragraphs) == 1:
                self._order.append(np.zeros((1, 1), dtype=np.int32))
                continue
            if force_answer:
                sample1 = [i for i, p in enumerate(q.paragraphs) if len(p.answer_spans) > 0]
            else:
                sample1 = list(range(len(q.paragraphs)))

            if (len(self.overample_first_answer) > 0 and
                    not (force_answer and len(sample1) == 1)):  # don't bother if there only is one answer
                ix = 0
                for i, p in enumerate(q.paragraphs):
                    if len(p.answer_spans) > 0:
                        sample1 += [i] * self.overample_first_answer[ix]
                        ix += 1
                        if ix >= len(self.overample_first_answer):
                            break

            permutations = []
            for i in sample1:
                for j in range(len(q.paragraphs)):
                    if j != i:
                        permutations.append((i, j))
            permutations = np.array(permutations, dtype=np.int32)
            np.random.shuffle(permutations)
            self._order.append(permutations)

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
        return ParagraphAndQuestionSpec(None, max_q_len, max_c_len, None)

    def get_epoch(self):
        return self._build_expanded_batches(self.questions)

    def _build_expanded_batches(self, questions):
        out = []
        # Decide what paragraphs to use for each question
        for i, q in enumerate(questions):
            order = self._order[i]
            out.append(ParagraphSelection(q, order[self._on[i]]))
            self._on[i] += 1
            if self._on[i] == len(order):
                self._on[i] = 0
                np.random.shuffle(order)

        # Sort by context length
        out.sort(key=lambda x: x.n_context_words)

        # Yield the correct batches
        group = 0
        for selection_batch in self.batcher.get_epoch(out):
            batch = []
            for selected in selection_batch:
                q = selected.question
                if self.merge:
                    paras = [q.paragraphs[i] for i in selected.selection]
                    # Sort paragraph by reading order, not rank order
                    paras.sort(key=lambda x: x.get_order())
                    answer_spans = []
                    text = []
                    for para in paras:
                        answer_spans.append(len(text) + para.answer_spans)
                        text += para.text
                    batch.append(ParagraphAndQuestion(text, q.question,
                                                      TokenSpans(q.answer_text, np.concatenate(answer_spans)),
                                                      q.question_id))
                else:
                    for i in selected.selection:
                        para = q.paragraphs[i]
                        batch.append(para.build_qa_pair(q.question, q.question_id, q.answer_text, group))
                    group += 1
            yield batch

    def get_samples(self, n_examples):
        n_batches = self.batcher.epoch_size(n_examples)
        return self._build_expanded_batches(np.random.choice(self.questions, n_examples, replace=False)), n_batches

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))


def multi_paragraph_word_counts(data):
    wc = Counter()
    for point in data:
        wc.update(point.question)
        for para in point.paragraphs:
            wc.update(para.get_context())
    return WordCounts(wc)


class IndividualParagraphBuilder(DatasetBuilder):
    """ Treat each paragraph as its own training point """

    def __init__(self, batcher: ListBatcher, force_answer: float):
        self.batcher = batcher
        self.force_answer = force_answer

    def build_stats(self, data: FilteredData):
        return multi_paragraph_word_counts(data.data)

    def build_dataset(self, data: FilteredData, corpus) -> Dataset:
        flattened = []
        for point in data.data:
            for para in point.paragraphs:
                flattened.append(ParagraphAndQuestion(para.text, point.question,
                                            TokenSpans(point.answer_text, para.answer_spans),
                                            point.question_id))
        return ParagraphAndQuestionDataset(flattened, self.batcher)


class RandomParagraphsBuilder(DatasetBuilder):
    def __init__(self, batching: ListBatcher, force_answer: float, n_to_sample=1):
        self.batching = batching
        self.force_answer = force_answer
        self.n_to_sample = n_to_sample

    def build_stats(self, data: Union[FilteredData, List]):
        if isinstance(data, FilteredData):
            return multi_paragraph_word_counts(data.data)
        else:
            return multi_paragraph_word_counts(data)

    def build_dataset(self, data: Union[FilteredData, List], corpus) -> Dataset:
        if isinstance(data, FilteredData):
            l = data.true_len
            data = data.data
        else:
            l = len(data)
        return RandomParagraphDataset(data, self.force_answer, l,
                                      self.n_to_sample, self.batching)


class StratifyParagraphsBuilder(DatasetBuilder):
    def __init__(self, batcher: ListBatcher,  oversample_answers: Union[int, List[int]],
                 only_answers: bool=False):
        self.batcher = batcher
        self.oversample_answers = oversample_answers
        self.only_answers = only_answers

    def build_dataset(self, data, evidence) -> Dataset:
        if isinstance(data, FilteredData):
            l = data.true_len
            data = data.data
        else:
            l = len(data)
        if self.only_answers:
            for q in data:
                q.paragraphs = [x for x in q.paragraphs if len(x.answer_spans) > 0]
            data = [x for x in data if len(x.paragraphs) > 0]

        if isinstance(self.oversample_answers, int):
            ov = [self.oversample_answers]
        else:
            ov = self.oversample_answers
        return StratifyParagraphsDataset(data, l, ov, self.batcher)

    @property
    def version(self):
        # Changed how sampling works
        return 2

    def build_stats(self, data) -> object:
        if isinstance(data, FilteredData):
            return multi_paragraph_word_counts(data.data)
        else:
            return multi_paragraph_word_counts(data)

    def __setstate__(self, state):
        if "only_answers" not in state:
            state["only_answers"] = False
        print(state)
        if state.get("oversample", 0) != 0:
            raise NotImplementedError()
        if "oversample_first" in state:
            state["oversample_answers"] = [state["oversample_first"]]
            del state["oversample_first"]

        super().__setstate__(state)


class RandomParagraphSetDatasetBuilder(DatasetBuilder):
    def __init__(self, batch_size: int, mode: str, force_answer: bool,
                 oversample_first_answer: Union[int, List[int]]):
        self.mode = mode
        self.oversample_first_answer = oversample_first_answer
        self.batch_size = batch_size
        self.force_answer = force_answer

    def build_stats(self, data: Union[FilteredData, List]):
        if isinstance(data, FilteredData):
            return multi_paragraph_word_counts(data.data)
        else:
            return multi_paragraph_word_counts(data)

    def build_dataset(self, data: Union[FilteredData, List], corpus) -> Dataset:
        if isinstance(data, FilteredData):
            l = data.true_len
            data = data.data
        else:
            l = len(data)
        if isinstance(self.oversample_first_answer, int):
            ov = [self.oversample_first_answer]
        else:
            ov = self.oversample_first_answer
        return RandomParagraphSetDataset(data, l, 2, self.batch_size, self.mode, self.force_answer, ov)


class StratifyParagraphSetsBuilder(DatasetBuilder):
    def __init__(self, batch_size: int, merge: bool, force_answer: bool,
                 oversample_first_answer: Union[int, List[int]]):
        self.batch_size = batch_size
        self.merge = merge
        self.force_answer = force_answer
        self.oversample_first_answer = oversample_first_answer

    def build_stats(self, data: Union[List, FilteredData]):
        if isinstance(data, FilteredData):
            return multi_paragraph_word_counts(data.data)
        else:
            return multi_paragraph_word_counts(data)

    def build_dataset(self, data: Union[FilteredData, List], corpus) -> Dataset:
        if isinstance(data, FilteredData):
            l = data.true_len
            data = data.data
        else:
            l = len(data)
        if isinstance(self.oversample_first_answer, int):
            ov = [self.oversample_first_answer]
        else:
            ov = self.oversample_first_answer
        return StratifiedParagraphSetDataset(data, l, self.batch_size, self.force_answer,
                                             ov, self.merge)

