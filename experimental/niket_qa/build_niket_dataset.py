import json
from os.path import join, isfile, exists, isdir
import pickle
from typing import List, Optional

from os import mkdir

from config import CORPUS_DIR, NIKET_QA
from configurable import Configurable
from data_processing.qa_training_data import ParagraphQaTrainingData, SentencesAndQuestion
from data_processing.span_data import TokenSpans
from data_processing.text_utils import NltkAndPunctTokenizer
from data_processing.word_vectors import load_word_vectors
from dataset import ListBatcher
from trivia_qa.answer_detection import FastNormalizedAnswerDetector
from utils import flatten_iterable, ResourceLoader
import re
import numpy as np


class NiketCorpus(Configurable):
    TRAIN_FILE = "train.pkl"
    DEV_FILE = "dev.pkl"
    TEST_FILE = "test.pkl"
    WORD_VEC_SUFFIX = "_pruned"

    @classmethod
    def build(cls, train, dev, test, version):
        base_dir = join(CORPUS_DIR, "niketv%d" % version)
        if not isdir(base_dir):
            mkdir(base_dir)
        with open(join(base_dir, cls.TRAIN_FILE), "wb") as f:
            pickle.dump(train, f)
        if dev is not None:
            with open(join(base_dir, cls.DEV_FILE), "wb") as f:
                pickle.dump(dev, f)
        if dev is not None:
            with open(join(base_dir, cls.TEST_FILE), "wb") as f:
                pickle.dump(test, f)

    def __init__(self, version=4):
        self.base_dir = join(CORPUS_DIR, "niketv%d" % version)

    def get_pruned_word_vecs(self, word_vec_name, voc=None):
        # TODO actually use this cache
        vec_file = join(self.base_dir, word_vec_name + self.WORD_VEC_SUFFIX + ".npy")
        if isfile(vec_file):
            print("Loading word vec %s for %s from cache" % (word_vec_name, self.name))
            with open(vec_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Building pruned word vec %s for %s" % (self.name, word_vec_name))
            voc = set()
            for q in self.get_train():
                voc.update(q.question)
                for c in q.context:
                    voc.update(c)
            vecs = load_word_vectors(word_vec_name, voc)
            with open(vec_file, "wb") as f:
                pickle.dump(vecs, f)
            return vecs

    def get_resource_loader(self):
        return ResourceLoader()

    def get_train(self):
        with open(join(self.base_dir, self.TRAIN_FILE), "rb") as f:
            return pickle.load(f)

    def get_dev(self):
        filename = join(self.base_dir, self.DEV_FILE)
        if not exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_test(self):
        filename = join(self.base_dir, self.TEST_FILE)
        if not exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)


class NiketTrainingData(ParagraphQaTrainingData):
    def __init__(self, percent_train_dev: Optional[float],
                 train_batcher: ListBatcher,
                 eval_batcher: ListBatcher,
                 prune_no_answers: bool):
        self.prune_no_answers = prune_no_answers
        super().__init__(NiketCorpus(), percent_train_dev, train_batcher, eval_batcher)

    def _preprocess(self, questions):
        # filter out no-answer questions
        if self.prune_no_answers:
            return [x for x in questions if len(x.answer.answer_spans) > 0], len(questions)
        print(len([x for x in questions if len(x.answer.answer_spans) > 0]))
        return questions, len(questions)


clean_start = re.compile("^(the|a|an|in)\s*\\b")
clean_end = re.compile("\s*\.$")


def find_answers(text: List[str], answers: List[str]):
    answers = [clean_start.sub("", x) for x in answers]
    answers = [clean_end.sub("", x) for x in answers]
    occurances = []
    for ans in answers:
        if len(ans) == 1:
            return []
        ans = ans.replace(" ", "")
        for start, word in enumerate(text):
            if not ans.startswith(word):
                if word.startswith(ans):
                    occurances.append((start, start))
                continue
            context_str = word
            end = start + 1
            while len(context_str) < len(ans) and end < len(text):
                context_str += text[end]
                end += 1
            # allow a slack of one char for "s" or "." in particular
            if context_str == ans or context_str[:-1] == ans:
                occurances.append((start, end-1))
    return list(set(occurances))


def build(source_file):
    tokenizer = NltkAndPunctTokenizer()

    with open(source_file, "r") as f:
        data = json.load(f)

    questions = []
    data = data["data"]
    para_id = 0
    for article in data:
        for paragraph in article['paragraphs']:
            context = tokenizer.tokenize_paragraph(paragraph["context"])
            flat_context = flatten_iterable(context)
            for question in paragraph["qas"]:
                answers = [ans["text"] for ans in question["answers"]]
                occ = find_answers(flat_context, answers)

                print()
                print(answers)
                if answers == ['DUMMY']:
                    answers = []

                tmp = list(flat_context)
                for s,e in occ:
                    tmp[s] = "{{{" + tmp[s]
                    tmp[e] = tmp[e] + "}}}"
                print(" ".join(tmp))

                if len(occ) == 0:
                    occ = np.zeros([0, 2], dtype=np.int32)
                else:
                    occ = np.array(occ, dtype=np.int32)
                questions.append(SentencesAndQuestion(context, tokenizer.tokenize_sentence(question["question"]),
                                                      TokenSpans(answers, occ), question["id"]))

            para_id += 1
    return questions


def main(version):
    base = join(NIKET_QA, "v%d" % version)
    train_questions = build(join(base, "bidafv%d_train.json" % version))

    dev_question = build(join(base, "bidafv%d_dev.json" % version))

    test_questions = build(join(base, "bidafv%d_test.json" % version))

    NiketCorpus.build(train_questions, dev_question, test_questions, version)
    print("Done!")


def count():
    data = NiketCorpus()
    l = np.array(flatten_iterable([(e-s+1) for s,e in x.answer.answer_spans] for x  in data.get_train()))
    print(l.mean())
    print((l == 1).mean())


if __name__ == "__main__":
    main(4)

