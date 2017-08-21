import unicodedata

import sys

from utils import ResourceLoader

try:
  import ujson as json
except ImportError:
  import json

import pickle
import re
from os import mkdir

from os.path import join, exists

from itertools import islice
from typing import List, Optional

from config import CORPUS_DIR, TRIVIA_QA, TRIVIA_QA_UNFILTERED
from configurable import Configurable
from trivia_qa.answer_detection import compute_answer_spans_par, FastNormalizedAnswerDetector
from trivia_qa.evidence_corpus import build_tokenized_corpus, TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import iter_trivia_question, TriviaQaQuestion


"""
Build span-level training data from the raw trivia-qa inputs, in particular annotates each question/doc
with the places the question answer's occur within the document and saves the resulting dataset
"""


def build_dataset(name, tokenizer, train_files,
                  answer_detector, n_process, prune_unmapped_docs=True,
                  sample=None):
    out_dir = join(CORPUS_DIR, "triviaqa", name)
    if not exists(out_dir):
        mkdir(out_dir)

    file_map = {}

    for name, filename in train_files.items():
        print("Loading %s questions" % name)
        if sample is None:
            questions = list(iter_trivia_question(filename, file_map, False))
        else:
            if isinstance(sample,  int):
                questions = list(islice(iter_trivia_question(filename, file_map, False), sample))
            elif isinstance(sample, dict):
                questions = list(islice(iter_trivia_question(filename, file_map, False), sample[name]))
            else:
                raise ValueError()

        if prune_unmapped_docs:
            for q in questions:
                if q.web_docs is not None:
                    q.web_docs = [x for x in q.web_docs if x.doc_id in file_map]
                q.entity_docs = [x for x in q.entity_docs if x.doc_id in file_map]

        print("Adding answers for %s question" % name)
        corpus = TriviaQaEvidenceCorpusTxt(file_map)
        questions = compute_answer_spans_par(questions, corpus, tokenizer, answer_detector, n_process)
        for q in questions:  # Sanity check, we should have answers for everything
            for doc in q.all_docs:
                if doc.doc_id in file_map:
                    if doc.answer_spans is None:
                        raise ValueError()

        print("Saving %s question" % name)
        with open(join(out_dir, name + ".pkl"), "wb") as f:
            pickle.dump(questions, f)

    print("Dumping file mapping")
    with open(join(out_dir, "file_map.json"), "w") as f:
        json.dump(file_map, f)

    print("Complete")


class TriviaQaSpanCorpus(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.dir = join(CORPUS_DIR, "triviaqa", corpus_name)
        with open(join(self.dir, "file_map.json"), "r") as f:
            file_map = json.load(f)
        for k, v in file_map.items():
            # We need have a consistend unicode format for the filenames,
            # so we always keep filenames in NFD format
            file_map[k] = unicodedata.normalize("NFD", v)
        self.evidence = TriviaQaEvidenceCorpusTxt(file_map)

    def get_train(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "train.pkl"), "rb") as f:
            return pickle.load(f)

    def get_dev(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "dev.pkl"), "rb") as f:
            return pickle.load(f)

    def get_verified(self) -> Optional[List[TriviaQaQuestion]]:
        verified_dir = join(self.dir, "verified.pkl")
        if not exists(verified_dir):
            return None
        with open(verified_dir, "rb") as f:
            return pickle.load(f)

    def get_resource_loader(self):
        return ResourceLoader()

    @property
    def name(self):
        return self.corpus_name


class TriviaQaWebDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web")


class TriviaQaOpenDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web-open")


class TriviaQaSampleWebDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web-sample")


def build_wiki_corpus():
    build_dataset("wiki", "NLTK_AND_CLEAN",
                  dict(
                      verified=join(TRIVIA_QA, "qa", "verified-wikipedia-dev.json"),
                      dev=join(TRIVIA_QA, "qa", "wikipedia-dev.json"),
                      train=join(TRIVIA_QA, "qa", "wikipedia-train.json"),
                  ),
                  FastNormalizedAnswerDetector(), 2)


def build_web_corpus():
    build_dataset("web", "NLTK_AND_CLEAN",
                  dict(
                      verified=join(TRIVIA_QA, "qa", "verified-web-dev.json"),
                      dev=join(TRIVIA_QA, "qa", "web-dev.json"),
                      train=join(TRIVIA_QA, "qa", "web-train.json"),
                  ),
                  FastNormalizedAnswerDetector(), 2)


def build_sample_corpus():
    build_dataset("web-sample", "NLTK_AND_CLEAN",
                  dict(
                      dev=join(TRIVIA_QA, "qa", "web-dev.json"),
                      train=join(TRIVIA_QA, "qa", "web-train.json"),
                  ),
                  FastNormalizedAnswerDetector(), 2, sample=1000)


def build_unfiltered_corpus():
    build_dataset("web-open", "NLTK_AND_CLEAN",
                  dict(
                      dev=join(TRIVIA_QA_UNFILTERED, "unfiltered-web-dev.json"),
                      train=join(TRIVIA_QA_UNFILTERED, "unfiltered-web-train.json"),
                  ),
                  answer_detector=FastNormalizedAnswerDetector(),
                  n_process=2)


if __name__ == "__main__":
    build_sample_corpus()
    # build_unfiltered_corpus()


