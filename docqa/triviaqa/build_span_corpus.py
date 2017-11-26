import argparse
import json
import pickle
import unicodedata
from itertools import islice
from os import mkdir
from os.path import join, exists
from typing import List, Optional, Dict

from docqa.config import CORPUS_DIR, TRIVIA_QA, TRIVIA_QA_UNFILTERED
from docqa.configurable import Configurable
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.triviaqa.answer_detection import compute_answer_spans_par, FastNormalizedAnswerDetector
from docqa.triviaqa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from docqa.triviaqa.read_data import iter_trivia_question, TriviaQaQuestion
from docqa.utils import ResourceLoader

"""
Build span-level training data from the raw trivia-qa inputs, in particular load the questions
from the json file and annotates each question/doc with the places the question answer's occur 
within the document, and save the results in our format. Assumes the evidence corpus has 
already been preprocessed 
"""


def build_dataset(name: str, tokenizer, train_files: Dict[str, str],
                  answer_detector, n_process: int, prune_unmapped_docs=True,
                  sample=None):
    out_dir = join(CORPUS_DIR, "triviaqa", name)
    if not exists(out_dir):
        mkdir(out_dir)

    file_map = {}  # maps document_id -> filename

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
        for q in questions:  # Sanity check, we should have answers for everything (even if of size 0)
            if q.answer is None:
                continue
            for doc in q.all_docs:
                if doc.doc_id in file_map:
                    if doc.answer_spans is None:
                        raise RuntimeError()

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
            file_map[k] = unicodedata.normalize("NFD", v)
        self.evidence = TriviaQaEvidenceCorpusTxt(file_map)

    def get_train(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "train.pkl"), "rb") as f:
            return pickle.load(f)

    def get_dev(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "dev.pkl"), "rb") as f:
            return pickle.load(f)

    def get_test(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "test.pkl"), "rb") as f:
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


class TriviaQaWikiDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("wiki")


class TriviaQaOpenDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web-open")


class TriviaQaSampleWebDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web-sample")


def build_wiki_corpus(n_processes):
    build_dataset("wiki", NltkAndPunctTokenizer(),
                  dict(
                      verified=join(TRIVIA_QA, "qa", "verified-wikipedia-dev.json"),
                      dev=join(TRIVIA_QA, "qa", "wikipedia-dev.json"),
                      train=join(TRIVIA_QA, "qa", "wikipedia-train.json"),
                      test=join(TRIVIA_QA, "qa", "wikipedia-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes)


def build_web_corpus(n_processes):
    build_dataset("web", NltkAndPunctTokenizer(),
                  dict(
                      verified=join(TRIVIA_QA, "qa", "verified-web-dev.json"),
                      dev=join(TRIVIA_QA, "qa", "web-dev.json"),
                      train=join(TRIVIA_QA, "qa", "web-train.json"),
                      test=join(TRIVIA_QA, "qa", "web-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes)


def build_sample_corpus(n_processes):
    build_dataset("web-sample", NltkAndPunctTokenizer(),
                  dict(
                      dev=join(TRIVIA_QA, "qa", "web-dev.json"),
                      train=join(TRIVIA_QA, "qa", "web-train.json"),
                  ),
                  FastNormalizedAnswerDetector(), n_processes, sample=1000)


def build_unfiltered_corpus(n_processes):
    build_dataset("web-open", NltkAndPunctTokenizer(),
                  dict(
                      dev=join(TRIVIA_QA_UNFILTERED, "unfiltered-web-dev.json"),
                      train=join(TRIVIA_QA_UNFILTERED, "unfiltered-web-train.json"),
                      test=join(TRIVIA_QA_UNFILTERED, "unfiltered-web-test-without-answers.json")
                  ),
                  answer_detector=FastNormalizedAnswerDetector(),
                  n_process=n_processes)


def main():
    parser = argparse.ArgumentParser("Pre-procsess TriviaQA data")
    parser.add_argument("corpus", choices=["web", "wiki", "web-open"])
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()
    if args.corpus == "web":
        build_web_corpus(args.n_processes)
    elif args.corpus == "wiki":
        build_wiki_corpus(args.n_processes)
    elif args.corpus == "web-open":
        build_unfiltered_corpus(args.n_processes)
    else:
        raise RuntimeError()


if __name__ == "__main__":
    main()


