import json
import re
import string
import pickle
import time

from os.path import exists

from data_processing.text_utils import get_paragraph_tokenizer
from trivia_qa.read_data import iter_trivia_question
from trivia_qa.trivia_qa_eval import normalize_answer, f1_score
from utils import flatten_iterable, group, split

import numpy as np


"""
Tools for turning the aliases and answer strings from TriviaQa into labelled spans
"""


class NormalizedAnswerDetector(object):
    """ Try to labels tokens sequences, such that the extracted sequence would be evaluated as 100% correct
    by the official trivia-qa evaluation script"""
    def __init__(self):
        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        words = [normalize_answer(w) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next == "":
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


class FastNormalizedAnswerDetector(object):
    """ almost twice as fast and very,very close to NormalizedAnswerDetector's output """

    def __init__(self):
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        words = [w.lower().strip(self.strip) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next in self.skip:
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


class CarefulAnswerDetector(object):
    """
    There are some common false negatives in the above answer detection, in particular plurals of answers are
    often not found (nor are counted correct by the official script). This detector makes a stronger effort to
    find them, although its unclear if training with these additional answers would hurt/help our overall score
    """
    def __init__(self):
        self.skip = {"a", "an", "the", "&", "and", "-", "\u2019", "\u2018", "\"", ";", "'",
                     "(", ")", "'s'", "s", ":", ",", "."}
        self.answer_regex = None
        self.aliases = None

    def set_question(self, normalized_aliases):
        answer_regex = []
        self.aliases = normalized_aliases
        for answer in normalized_aliases:
            tokens = []
            for token in answer:
                if len(token) > 1:
                    tokens.append(token + "s?")
                else:
                    tokens.append(token)
            if tokens[-1] == "s":
                tokens[-1] = "s?"

            answer_regex.append([re.compile(x, re.IGNORECASE) for x in tokens])

        self.answer_regex = answer_regex

    def any_found(self, para):
        words = flatten_iterable(para)
        occurances = []
        for answer_ix, answer in enumerate(self.answer_regex):
            word_starts = [i for i, w in enumerate(words) if answer[0].fullmatch(w)]
            n_tokens = len(answer)
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token].match(next):
                        ans_token += 1
                        end += 1
                    elif next in self.skip:
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


def evaluate_question_detector(questions, corpus, word_tokenize, detector, reference_detector=None, compute_f1s=False):
    n_no_docs = 0
    answer_per_doc = []
    answer_f1s = []

    for question_ix, q in enumerate(questions):
        q.question = word_tokenize(q.question)
        tokenized_aliases = [word_tokenize(x) for x in q.answer.normalized_aliases]
        detector.set_question(tokenized_aliases)

        for doc in q.all_docs:
            doc = corpus.get_document(doc.doc_id)
            if doc is None:
                n_no_docs += 1
                continue

            output = []
            for i, para in enumerate(doc):
                for s,e in detector.any_found(para):
                    output.append((i, s, e))

            if len(output) == 0 and reference_detector is not None:
                if reference_detector is not None:
                    reference_detector.set_question(tokenized_aliases)
                    detected = []
                    for i, para in enumerate(doc):
                        for s, e in reference_detector.any_found(para):
                            detected.append((i, s, e))

                    if len(detected) > 0:
                        print("Found a difference")
                        print(q.answer.normalized_aliases)
                        print(tokenized_aliases)
                        for p, s, e in detected:
                            token = flatten_iterable(doc[p])[s:e]
                            print(token)

            answer_per_doc.append(output)

            if compute_f1s:
                f1s = []
                for p, s, e in output:
                    token = flatten_iterable(doc[p])[s:e]
                    answer = normalize_answer(" ".join(token))
                    f1 = 0
                    for gt in q.answer.normalized_aliases:
                        f1 = max(f1, f1_score(answer, gt))
                    f1s.append(f1)
                answer_f1s.append(f1s)

    n_answers = sum(len(x) for x in answer_per_doc)
    print("Found %d answers (av %.4f)" % (n_answers, n_answers/len(answer_per_doc)))
    print("%.4f docs have answers" % np.mean([len(x) > 0 for x in answer_per_doc]))
    print("Average f1 is %.4f" % np.mean(flatten_iterable(answer_f1s)))


def compute_answer_spans(questions, corpus, word_tokenize, detector):
    for i, q in enumerate(questions):
        if i % 500 == 0:
            print("Completed question %d of %d (%.3f)" % (i, len(questions), i/len(questions)))
        tokenized_aliases = [word_tokenize(x) for x in q.answer.all_answers]
        if len(tokenized_aliases) == 0:
            raise ValueError()
        detector.set_question(tokenized_aliases)
        q.question = word_tokenize(q.question)
        for doc in q.all_docs:
            text = corpus.get_document(doc.doc_id)
            if text is None:
                raise ValueError()
            spans = []
            offset = 0
            for para_ix, para in enumerate(text):
                for s, e in detector.any_found(para):
                    spans.append((s+offset, e+offset-1))  # turn into inclusive span
                offset += sum(len(s) for s in para)
            if len(spans) == 0:
                spans = np.zeros((0, 2), dtype=np.int32)
            else:
                spans = np.array(spans, dtype=np.int32)
            doc.answer_spans = spans


def _compute_answer_spans_chunk(questions, corpus, tokenizer, detector):
    word_tokenize = get_paragraph_tokenizer(tokenizer)[1]
    compute_answer_spans(questions, corpus, word_tokenize, detector)
    return questions


def compute_answer_spans_par(questions, corpus, tokenizer, detector, n_processes):
    if n_processes == 1:
        word_tokenize = get_paragraph_tokenizer(tokenizer)[1]
        compute_answer_spans(questions, corpus, word_tokenize, detector)
        return questions
    from multiprocessing import Pool
    p = Pool(n_processes)
    chunks = split(questions, n_processes)
    questions = flatten_iterable(p.starmap(_compute_answer_spans_chunk, [[c, corpus, tokenizer, detector] for c in chunks]))
    return questions


def get_sample():
    cache = "/tmp/sample.pkl"
    if exists(cache):
        print("Loading from cache")
        with open(cache, "rb") as f:
            questions, file_map = pickle.load(f)
            return questions, file_map

    file_map = {}
    questions = list(iter_trivia_question("/Users/chrisc/Programming/data/trivia-qa/qa/web-train.json", file_map, True))
    for q in questions:
        for doc in q.web_docs:
            doc.trivia_qa_selected = True
        for doc in q.entity_docs:
            doc.trivia_qa_selected = True

    np.random.shuffle(questions)
    questions = questions[:6000]
    with open(cache, "wb") as f:
        pickle.dump((list(questions), file_map), f)
    return questions, file_map


