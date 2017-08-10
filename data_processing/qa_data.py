from collections import Counter
from typing import List, Union, Optional, Set, Tuple, Dict, Callable

import numpy as np
from configurable import Configurable

from dataset import Dataset, TrainingData, ListDataset, ListBatcher
from utils import ResourceLoader, flatten_iterable, max_or_none

"""
Objects to represent question-context-answer training points, relating utility methods, 
and objects to use them as part of a `Dataset` that can be used to train models.
"""


class Answer(object):
    """ Abstract representation of an answer to question """

    def get_vocab(self):
        raise NotImplemented()

    def to_json_object(self):
        raise NotImplemented()


class ParagraphAndQuestion(object):
    """
    This is our standard unit of training data, list of sentences as context and a single question/answer.
    The answer type is unspecified and depends on the application.
    """

    def __init__(self, context: List[List[str]], question: List[str],
                 answer: Optional[Answer], question_id: object):
        self.context = context
        self.question = question
        self.answer = answer
        self.question_id = question_id

    @property
    def n_context_words(self):
        return sum(len(s) for s in self.context)


class ParagraphAndQuestionSpec(object):
    """ Bound on the size of `ParagraphAndQuestion` objects """
    def __init__(self, batch_size, max_question_words, max_num_context_words,
                 max_num_sents, max_sent_size, max_word_size):
        self.batch_size = batch_size
        self.max_num_quesiton_words = max_question_words
        self.max_num_context_words = max_num_context_words
        self.max_num_sents = max_num_sents
        self.max_context_sent_size = max_sent_size
        self.max_word_size = max_word_size

    def __add__(self, o):
        return ParagraphAndQuestionSpec(
            max_or_none(self.batch_size, o.batch_size),
            max_or_none(self.max_num_quesiton_words, o.max_num_quesiton_words),
            max_or_none(self.max_num_context_words, o.max_num_context_words),
            max_or_none(self.max_num_sents, o.max_num_sents),
            max_or_none(self.max_context_sent_size, o.max_context_sent_size),
            max_or_none(self.max_word_size, o.max_word_size)
        )


def get_input_spec(batch_size, data: List[ParagraphAndQuestion]) -> ParagraphAndQuestionSpec:
    max_num_sents = 0
    max_sent_size = 0
    max_ques_size = 0
    max_word_size = 0
    max_para_size = 0
    for data_point in data:
        max_num_sents = max(max_num_sents, len(data_point.context))
        max_sent_size = max(max_sent_size, max(len(s) for s in data_point.context))
        max_word_size = max(max_word_size, max(len(word) for sent in data_point.context for word in sent))
        max_para_size = max(max_para_size, sum(len(sent) for sent in data_point.context))
        if data_point.question is not None:
            max_ques_size = max(max_ques_size, len(data_point.question))
            max_word_size = max(max_word_size, max(len(word) for word in data_point.question))
    return ParagraphAndQuestionSpec(batch_size, max_ques_size, max_para_size,
                                    max_num_sents, max_sent_size, max_word_size)


class QaCorpusStats(object):
    def __init__(self, question_counts, context_counts, unique_context_counts, special_tokens=None):
        self.question_counts = question_counts
        self.context_counts = context_counts
        self.unique_context_counts = unique_context_counts
        self.special_tokens = special_tokens

    def get_question_counts(self):
        return self.question_counts

    def get_context_counts(self):
        return self.context_counts

    def get_document_counts(self):
        return self.unique_context_counts

    def get_word_counts(self):
        return self.context_counts + self.question_counts


class QaCorpusLazyStats(object):
    def __init__(self, data: List[ParagraphAndQuestion], special_tokens=None):
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
                for sent in q.context:
                    counter.update(sent)
            self._context_counts = counter
        return self._context_counts

    def get_word_counts(self):
        return self.get_context_counts() + self.get_question_counts()


def compute_voc(data: List[ParagraphAndQuestion]):
    voc = set()
    for point in data:
        voc.update(point.question)
        if point.answer is not None:
            voc.update(point.answer.get_vocab())
        for sent in point.context:
            voc.update(sent)
    return voc


class ParagraphQuestionFilter(Configurable):
    def keep(self, data_point: ParagraphAndQuestion) -> bool:
        raise NotImplemented()


class QuestionFilter(ParagraphQuestionFilter):
    def __init__(self, ques_size_th: int):
        self.ques_size_th = ques_size_th

    def keep(self, data_point: ParagraphAndQuestion):
        return len(data_point.question) <= self.ques_size_th


class AnswerWord(ParagraphQuestionFilter):
    def __init__(self, para_size_th: int):
        self.para_size_th = para_size_th

    def keep(self, data_point: ParagraphAndQuestion):
        return all(ans.para_word_end < self.para_size_th for ans in data_point.answer)


class AnySplitAnswerFilter(ParagraphQuestionFilter):
    def keep(self, data_point: ParagraphAndQuestion):
        for answer in data_point.answer:
            if answer.sent_start != answer.sent_end:
                return False
        return True


class AnswerSentence(ParagraphQuestionFilter):
    def __init__(self, sent_size_th: Union[int, None] = None,
                 num_sent_th: Union[int, None]=None):
        self.num_sent_th = num_sent_th
        self.sent_size_th = sent_size_th

    def keep(self, data_point: ParagraphAndQuestion):
        for answer in data_point.answer:
            if self.num_sent_th is not None:
                if answer.sent_end >= self.num_sent_th:
                    return False
            if self.sent_size_th is not None:
                if answer.word_start >= self.sent_size_th:
                    return False
        return True


def apply_filters(data: List, data_filters: List[ParagraphQuestionFilter], name: str):
    if len(data) == 0:
        raise ValueError()
    if len(data_filters) == 0:
        return data
    else:
        pruned = []
        removed = np.zeros(len(data_filters), dtype=np.int32)
        for x in data:
            keep = True
            for i,f in enumerate(data_filters):
                if not f.keep(x):
                    keep = False
                    removed[i] += 1
                    break
            if keep:
                pruned.append(x)
        for i,x in enumerate(data_filters):
            print("\t%s filtered %d(%.5f) from %s" % (x.__class__.__name__, removed[i], removed[i]/len(data), name))
        n_removed = len(data)-len(pruned)
        print("Pruned a total of %d/%d (%.3f) for %s" % (n_removed, len(data), n_removed/len(data), name))
        return pruned


def question_sorter(cluster):
    if cluster == "context_words":
        cluster = lambda x: sum(len(s) for s in x.context)
    elif cluster is not None and cluster.startswith("bucket_context_words_"):
        bucket_size = int(cluster[cluster.rfind("_") + 1:])
        cluster = lambda x: sum(len(s) for s in x.context) // bucket_size
    elif cluster == "sentence_len":
        cluster = lambda x: max(len(s) for s in x.context)
    elif cluster is not None:
        raise ValueError()
    else:
        return None
    return cluster


class ParagraphAndQuestionDataset(ListDataset):
    def get_spec(self) -> ParagraphAndQuestionSpec:
        return get_input_spec(self.batching.get_fixed_batch_size(), self.data)

    def get_vocab(self) -> Set[str]:
        return compute_voc(self.data)


class QaCorpus(Configurable):
    def get_train_corpus(self):
        raise NotImplemented()

    def get_train(self) -> List[ParagraphAndQuestion]:
        raise NotImplemented()

    def get_dev(self) -> List[ParagraphAndQuestion]:
        raise NotImplemented()

    def get_resource_loader(self) -> ResourceLoader:
        raise NotImplemented()


class FixedParagraphQaTrainingData(TrainingData):
    def __init__(self,
                 corpus: QaCorpus,
                 percent_train_dev: Optional[float],
                 train_batcher: ListBatcher,
                 eval_batcher: ListBatcher,
                 data_filters: List[ParagraphQuestionFilter] = None):
        self.percent_train_dev = percent_train_dev
        self.eval_batcher = eval_batcher
        self.train_batcher = train_batcher
        self.corpus = corpus
        self.data_filters = data_filters
        self._train_corpus = None
        self._train = None
        self._dev = None

    @property
    def name(self):
        return self.corpus.name

    def _load_data(self):
        if self._train is not None:
            return
        print("Loading data for: " + self.corpus.name)
        self._train, self._train_corpus = self.corpus.get_train_corpus()
        if self.percent_train_dev is None:
            self._dev = self.corpus.get_dev()
        else:
            raise NotImplemented()
        if self.data_filters is not None:
            self._dev = apply_filters(self._dev, self.data_filters, "dev")
            self._train = apply_filters(self._train, self.data_filters, "train")

    def get_train(self) -> Dataset:
        self._load_data()
        return ParagraphAndQuestionDataset(self._train, self.train_batcher)

    def get_train_corpus(self):
        self._load_data()
        return self._train_corpus

    def get_eval(self) -> Dict[str, Dataset]:
        self._load_data()
        return dict(dev=ParagraphAndQuestionDataset(self._dev, self.eval_batcher),
                    train=ParagraphAndQuestionDataset(self._train, self.eval_batcher))

    def get_resource_loader(self) -> ResourceLoader:
        return self.corpus.get_resource_loader()

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        state["_train_corpus"] = None
        return state

    def __setstate__(self, state):
        if "batching" in state:
            raise NotImplementedError("Backwards compatability")
        self.__dict__ = state


# Keep around to keep pickle happy for legacy cases
class Batcher(Configurable):
    def __init__(self):
        raise ValueError("Deprecated!")
