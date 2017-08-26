import gzip
import pickle
from collections import Counter
from threading import Lock
from typing import List, Dict, Iterable, Union, Tuple, Optional

import numpy as np
from tqdm import tqdm

from configurable import Configurable
from data_processing.text_utils import NltkPlusStopWords
from dataset import TrainingData, Dataset, ListDataset
from trivia_qa.read_data import TriviaQaQuestion
from utils import split, flatten_iterable, group, ResourceLoader


class Preprocessor(Configurable):
    def preprocess(self, question: Iterable, evidence) -> object:
        """ Map elements to an unspecified intermediate format """
        raise NotImplementedError()

    def finalize(self, x):
        pass


class DatasetBuilder(Configurable):

    def build_dataset(self, data, evidence, is_train: bool) -> Dataset:
        """ Map the intermeidate format to a Dataset object """
        raise NotImplementedError()

    def build_stats(self, data) -> object:
        """ Map the intermeidate format to corpus statistic object, as will used in `TrainingData` """
        raise NotImplementedError()


class LazyCorpusStatistics(object):
    def __init__(self, data: List, special_tokens=None):
        self.data = data
        self.special_tokens = special_tokens

    def get_word_counts(self):
        counts = Counter()
        for point in self.data:
            counts.update(point.get_text())
        return counts


class FilteredData(object):
    def __init__(self, data: List, true_len: int):
        self.data = data
        self.true_len = true_len

    def __add__(self, other):
        return FilteredData(self.data + other.data, self.true_len + other.true_len)


def _preprocess_and_count(questions: List[TriviaQaQuestion], evidence, preprocessor: Preprocessor):
    count = len(questions)
    output = preprocessor.preprocess(questions, evidence)
    return output, count


def preprocess_par(questions: List[TriviaQaQuestion], evidence, preprocessor,
                   n_processes=2, chunk_size=200, name=None):
    if n_processes is None or chunk_size is None or\
                    n_processes <= 0 or chunk_size <= 0:
        raise ValueError()

    if n_processes == 1:
        out = preprocessor.preprocess(tqdm(questions, desc=name, ncols=80), evidence)
        preprocessor.finalize(out)
        return out
    else:
        from multiprocessing import Pool
        chunks = split(questions, n_processes)
        chunks = flatten_iterable([group(c, chunk_size) for c in chunks])
        print("Processing %d chunks with %d processes" % (len(chunks), n_processes))
        pbar = tqdm(total=len(questions), desc=name, ncols=80)
        lock = Lock()

        def call_back(results):
            preprocessor.finalize(results[0])
            with lock:  # FIXME Even with the lock, the progress bar still is jumping around
                pbar.update(results[1])

        with Pool(n_processes) as pool:
            results = [pool.apply_async(_preprocess_and_count, [c, evidence, preprocessor], callback=call_back)
                       for c in chunks]
            results = [r.get()[0] for r in results]

        pbar.close()
        output = results[0]
        for r in results[1:]:
            output += r
        return output


class PreprocessedData(TrainingData):

    def __init__(self,
                 corpus,
                 preprocesser: Optional[Preprocessor],
                 builder: DatasetBuilder,
                 eval_on_verified: bool=True,
                 eval_on_train: bool = True,
                 hold_out_train: Optional[Tuple[int, int]]= None,
                 sample=None, sample_dev=None):
        self.hold_out_train = hold_out_train
        self.eval_on_train = eval_on_train
        self.sample = sample
        self.eval_on_verified = eval_on_verified
        self.sample_dev = sample_dev
        self.corpus = corpus
        self.preprocesser = preprocesser
        self.builder = builder

        self._train = None
        self._dev = None
        self._verified_dev = None

    @property
    def name(self):
        return self.corpus.name

    def cache_preprocess(self, filename):
        # if self.sample is not None or self.sample_dev is not None or self.hold_out_train is not None:
        #     raise ValueError()
        if filename.endswith("gz"):
            handle = lambda a,b: gzip.open(a, b, compresslevel=3)
        else:
            handle = open
        with handle(filename, "wb") as f:
            pickle.dump([self.preprocesser, self._train, self._dev, self._verified_dev], f)

    def load_preprocess(self, filename):
        print("Loading preprocessed data...")
        if filename.endswith("gz"):
            handle = gzip.open
        else:
            handle = open
        with handle(filename, "rb") as f:
            stored = pickle.load(f)
            stored_preprocesser, self._train, self._dev, self._verified_dev = stored
        if stored_preprocesser.get_config() != self.preprocesser.get_config():
            # print("WARNING")
            raise ValueError()
        print("done")

    def preprocess(self, n_processes=1, chunk_size=500):
        if self._train is not None:
            return
        print("Loading data...")
        train_questions = self.corpus.get_train()
        if self.hold_out_train is not None:
            print("Using held out train")
            train_questions.sort(key=lambda q:q.question_id)
            np.random.RandomState(self.hold_out_train[0]).shuffle(train_questions)
            dev_questions = train_questions[:self.hold_out_train[1]]
            train_questions = train_questions[self.hold_out_train[1]:]
        else:
            dev_questions = self.corpus.get_dev()
        if self.eval_on_verified and hasattr(self.corpus, "get_verified"):  # TODO this is a bit hacky
            verified_questions = self.corpus.get_verified()
            if verified_questions is not None:
                # we don't eval on verified docs w/o any valid human answer
                verified_questions = [x for x in verified_questions if
                                      any(len(ans) > 0 for ans in x.answer.human_answers)]
        else:
            verified_questions = None

        if self.sample is not None:
            rng = np.random.RandomState(0)
            print("Warning: using sampling. Only should do this during debugging")
            train_questions = rng.choice(train_questions, self.sample, replace=False)
        if self.sample_dev is not None:
            rng = np.random.RandomState(0)
            print("Warning: using sampling. Only should do this during debugging")
            dev_questions = rng.choice(dev_questions, self.sample_dev, replace=False)

        if self.preprocesser:
            print("Preprocessing with %d processes..." % n_processes)
            out = []
            for name, questions in [("verified", verified_questions),
                                    ("dev", dev_questions),
                                    ("train", train_questions)]:
                if questions is None:
                    out.append(None)
                    continue
                data = preprocess_par(questions, self.corpus.evidence, self.preprocesser, n_processes, chunk_size, name)
                out.append(data)
            self._verified_dev, self._dev, self._train = out
        else:
            self._verified_dev, self._dev, self._train = verified_questions, dev_questions, train_questions

        print("Done")

    def _convert_questions(self, questions, n_processes, chunk_size):
        size = sum(len(q.all_docs) for q in questions)
        if size == 0:
            return []
        data = preprocess_par(questions, self.corpus.evidence, self.preprocesser, n_processes, chunk_size)
        return self.builder.build_dataset(data, self.corpus)

    def get_train(self) -> Dataset:
        return self.builder.build_dataset(self._train, self.corpus, True)

    def get_train_corpus(self):
        return self.builder.build_stats(self._train)

    def get_eval(self) -> Dict[str, Dataset]:
        corpus = self.corpus
        eval_set = dict(dev=self.builder.build_dataset(self._dev, corpus, False))
        if self.eval_on_train:
            eval_set["train"] = self.builder.build_dataset(self._train, corpus, False)
        if self.eval_on_verified:
            eval_set["verified-dev"] = self.builder.build_dataset(self._verified_dev, corpus, False)
        return eval_set

    def get_resource_loader(self) -> ResourceLoader:
        return self.corpus.get_resource_loader()

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_train"] = None
        state["_dev"] = None
        state["_verified_dev"] = None
        return state