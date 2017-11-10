from collections import Counter
from typing import List, Union, Optional, Set, Dict

import numpy as np
from docqa.dataset import Dataset, TrainingData, ListDataset, ListBatcher
from docqa.utils import ResourceLoader, flatten_iterable, max_or_none

from docqa.configurable import Configurable
from docqa.data_processing.preprocessed_corpus import DatasetBuilder, FilteredData

"""
Objects to represent question-context-answer training points / datasets
"""


class Answer(object):
    """ Abstract representation of an answer to question """
    def get_vocab(self):
        raise NotImplemented()


class ContextAndQuestion(object):
    """
    This is our standard unit of training data, context and a single question/answer.
    The answer type is unspecified and depends on the application.
    """

    def __init__(self, question: List[str], answer: Optional[Answer],
                 question_id: object, doc_id=None):
        self.question = question
        self.answer = answer
        self.question_id = question_id
        self.doc_id = doc_id

    @property
    def n_context_words(self) -> int:
        raise NotImplementedError()

    def get_context(self) -> List[str]:
        raise NotImplementedError()


class SentencesAndQuestion(ContextAndQuestion):

    def __init__(self, context: List[List[str]], question: List[str],
                 answer: Optional[Answer], question_id: object, doc_id=None):
        super().__init__(question, answer, question_id, doc_id)
        self.context = context

    @property
    def n_context_words(self):
        return sum(len(s) for s in self.context)

    def get_context(self):
        return flatten_iterable(self.context)


class ParagraphAndQuestion(ContextAndQuestion):

    def __init__(self, context: List[str], question: List[str],
                 answer: Optional[Answer], question_id: object, doc_id=None):
        super().__init__(question, answer, question_id, doc_id)
        self.context = context

    @property
    def n_context_words(self):
        return len(self.context)

    def get_context(self):
        return self.context


class ContextLenKey(Configurable):
    def __call__(self, q: ContextAndQuestion):
        return q.n_context_words


class ContextLenBucketedKey(Configurable):
    def __init__(self, bucket_size: int):
        self.bucket_size = bucket_size

    def __call__(self, q: SentencesAndQuestion):
        return q.n_context_words//self.bucket_size


class ParagraphAndQuestionSpec(object):
    """ Bound on the size of `ParagraphAndQuestion` objects """
    def __init__(self, batch_size, max_question_words=None,
                 max_num_context_words=None, max_word_size=None, max_batch_size=None):
        if batch_size is not None:
            if max_batch_size is None:
                max_batch_size = batch_size
            elif max_batch_size != batch_size:
                raise ValueError()
        self.batch_size = batch_size
        self.max_num_quesiton_words = max_question_words
        self.max_num_context_words = max_num_context_words
        self.max_word_size = max_word_size
        self.max_batch_size = max_batch_size

    def __add__(self, o):
        return ParagraphAndQuestionSpec(
            max_or_none(self.batch_size, o.batch_size),
            max_or_none(self.max_num_quesiton_words, o.max_num_quesiton_words),
            max_or_none(self.max_num_context_words, o.max_num_context_words),
            max_or_none(self.max_word_size, o.max_word_size),
            max_or_none(self.max_batch_size, o.max_batch_size)
        )


class QaCorpusLazyStats(object):
    """
    Metadata about a set of such questions we might need for things like
    computing which words vectors to use/train
    """

    def __init__(self, data: List[ContextAndQuestion]):
        self.data = data
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
                counter.update(q.get_context())
            self._context_counts = counter
        return self._context_counts

    def get_word_counts(self):
        return self.get_context_counts() + self.get_question_counts()


class QaCorpusStats(object):
    def __init__(self, question_counts, context_counts):
        self.question_counts = question_counts
        self.context_counts = context_counts

    def get_question_counts(self):
        return self.question_counts

    def get_context_counts(self):
        return self.context_counts

    def get_word_counts(self):
        return self.get_context_counts() + self.get_question_counts()


class WordCounts(object):
    def __init__(self, word_counts):
        self.word_counts = word_counts

    def get_word_counts(self):
        return self.word_counts


def compute_voc(data: List[ContextAndQuestion]):
    voc = set()
    for point in data:
        voc.update(point.question)
        if point.answer is not None:
            voc.update(point.answer.get_vocab())
        voc.update(point.get_context())
    return voc


class ParagraphQuestionFilter(Configurable):
    def keep(self, data_point: ContextAndQuestion) -> bool:
        raise NotImplemented()


class QuestionFilter(ParagraphQuestionFilter):
    def __init__(self, ques_size_th: int):
        self.ques_size_th = ques_size_th

    def keep(self, data_point: ContextAndQuestion):
        return len(data_point.question) <= self.ques_size_th


class AnswerWord(ParagraphQuestionFilter):
    def __init__(self, para_size_th: int):
        self.para_size_th = para_size_th

    def keep(self, data_point: ContextAndQuestion):
        return all(ans.para_word_end < self.para_size_th for ans in data_point.answer)


class AnySplitAnswerFilter(ParagraphQuestionFilter):
    def keep(self, data_point: SentencesAndQuestion):
        for answer in data_point.answer:
            if answer.sent_start != answer.sent_end:
                return False
        return True


class AnswerSentence(ParagraphQuestionFilter):
    def __init__(self, sent_size_th: Union[int, None] = None,
                 num_sent_th: Union[int, None]=None):
        self.num_sent_th = num_sent_th
        self.sent_size_th = sent_size_th

    def keep(self, data_point: SentencesAndQuestion):
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


def build_spec(batch_size: int,
               max_batch_size: int,
               data: List[ContextAndQuestion]) -> ParagraphAndQuestionSpec:
    max_ques_size = 0
    max_word_size = 0
    max_para_size = 0
    for data_point in data:
        context = data_point.get_context()
        max_word_size = max(max_word_size, max(len(word) for word in context))
        max_para_size = max(max_para_size, len(context))
        if data_point.question is not None:
            max_ques_size = max(max_ques_size, len(data_point.question))
            max_word_size = max(max_word_size, max(len(word) for word in data_point.question))
    return ParagraphAndQuestionSpec(batch_size, max_ques_size, max_para_size,
                                    max_word_size, max_batch_size)


class ParagraphAndQuestionDataset(ListDataset):
    """ Dataset with q/a pairs and that exposes some meta-data about its elements """
    def get_spec(self) -> ParagraphAndQuestionSpec:
        return build_spec(self.batching.get_fixed_batch_size(),
                          self.batching.get_max_batch_size(),
                          self.data)

    def get_vocab(self) -> Set[str]:
        return compute_voc(self.data)


class ParagraphAndQuestionsBuilder(DatasetBuilder):
    """ For use with the preprocesed_corpus framework """
    def __init__(self, batching: ListBatcher, sample=None, sample_seed=None):
        if sample_seed is not None and sample is None:
            raise ValueError("Seed set, but sampling not requested")
        self.batching = batching
        self.sample_seed = sample_seed
        self.sample = sample

    def build_stats(self, data):
        if isinstance(data, FilteredData):
            return QaCorpusLazyStats(data.data)
        else:
            return QaCorpusLazyStats(data)

    def build_dataset(self, data, evidence) -> Dataset:
        if isinstance(data, FilteredData):
            data, l = data.data, data.true_len
        else:
            data, l = data, None
        if self.sample is not None:
            cur_len = len(data)
            data = np.random.RandomState(self.sample_seed).choice(data, self.sample, replace=False)
            if l is not None:
                l *= len(data) / cur_len

        if l is None:
            l = len(data)
        print("Building dataset")
        print(len(data), l)
        return ParagraphAndQuestionDataset(data, self.batching, l)


class ParagraphQaTrainingData(TrainingData):
    """ Training data derived from a "corpus" objects loads elements and a preprocess method """

    def __init__(self,
                 corpus,
                 percent_train_dev: Optional[float],
                 train_batcher: ListBatcher,
                 eval_batcher: ListBatcher,
                 data_filters: List[ParagraphQuestionFilter] = None):
        self.percent_train_dev = percent_train_dev
        self.eval_batcher = eval_batcher
        self.train_batcher = train_batcher
        self.corpus = corpus
        self.data_filters = data_filters
        self._train = None
        self._dev = None
        self._dev_len = None
        self._train_len = None

    def _preprocess(self, x):
        return x, len(x)

    @property
    def name(self):
        return self.corpus.name

    def _load_data(self):
        if self._train is not None:
            return
        print("Loading data for: " + self.corpus.name)
        self._train, self._train_len = self._preprocess(self.corpus.get_train())
        if self.percent_train_dev is None:
            dev = self.corpus.get_dev()
            if dev is not None:
                self._dev, self._dev_len = self._preprocess(self.corpus.get_dev())
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
        return QaCorpusLazyStats(self._train)

    def get_eval(self) -> Dict[str, Dataset]:
        self._load_data()
        eval_sets = dict(train=ParagraphAndQuestionDataset(self._train, self.eval_batcher, self._train_len))
        if self._dev is not None:
            eval_sets["dev"] = ParagraphAndQuestionDataset(self._dev, self.eval_batcher, self._dev_len)
        return eval_sets

    def get_resource_loader(self) -> ResourceLoader:
        return self.corpus.get_resource_loader()

    def __getstate__(self):
        state = self.__dict__
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

