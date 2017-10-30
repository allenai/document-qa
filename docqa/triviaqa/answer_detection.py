import re
import string

import numpy as np
from tqdm import tqdm
from typing import List

from docqa.triviaqa.read_data import TriviaQaQuestion
from docqa.triviaqa.trivia_qa_eval import normalize_answer, f1_score
from docqa.utils import flatten_iterable, split

"""
Tools for turning the aliases and answer strings from TriviaQA into labelled spans
"""


class ExactMatchDetector(object):
    def __init__(self):
        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        words = [x.lower() for x in flatten_iterable(para)]
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
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


class NormalizedAnswerDetector(object):
    """ Try to labels tokens sequences, such that the extracted sequence would be evaluated as 100% correct
    by the official trivia-qa evaluation script """
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
        # These come from the TrivaQA official evaluation script
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        # Normalize the paragraph
        words = [w.lower().strip(self.strip) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            # Locations where the first word occurs
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)

            # Advance forward until we find all the words, skipping over articles
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
    since I never got around to trying it.
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
    """ Just for debugging """
    n_no_docs = 0
    answer_per_doc = []
    answer_f1s = []

    for question_ix, q in enumerate(tqdm(questions)):
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
    if len(answer_f1s) > 0:
        print("Average f1 is %.4f" % np.mean(flatten_iterable(answer_f1s)))


def compute_answer_spans(questions: List[TriviaQaQuestion], corpus, word_tokenize,
                         detector):

    for i, q in enumerate(questions):
        if i % 500 == 0:
            print("Completed question %d of %d (%.3f)" % (i, len(questions), i/len(questions)))
        q.question = word_tokenize(q.question)
        if q.answer is None:
            continue
        tokenized_aliases = [word_tokenize(x) for x in q.answer.all_answers]
        if len(tokenized_aliases) == 0:
            raise ValueError()
        detector.set_question(tokenized_aliases)
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
    # We use tokenize_paragraph since some questions can have multiple sentences,
    # but we still store the results as a flat list of tokens
    word_tokenize = tokenizer.tokenize_paragraph_flat
    compute_answer_spans(questions, corpus, word_tokenize, detector)
    return questions


def compute_answer_spans_par(questions: List[TriviaQaQuestion], corpus,
                             tokenizer, detector, n_processes: int):
    if n_processes == 1:
        word_tokenize = tokenizer.tokenize_paragraph_flat
        compute_answer_spans(questions, corpus, word_tokenize, detector)
        return questions
    from multiprocessing import Pool
    with Pool(n_processes) as p:
        chunks = split(questions, n_processes)
        questions = flatten_iterable(p.starmap(_compute_answer_spans_chunk,
                                               [[c, corpus, tokenizer, detector] for c in chunks]))
        return questions


def main():
    from trivia_qa.build_span_corpus import TriviaQaWebDataset
    from data_processing.text_utils import NltkAndPunctTokenizer

    dataset = TriviaQaWebDataset()
    qs = dataset.get_train()
    qs = np.random.RandomState(0).choice(qs, 1000, replace=False)
    evaluate_question_detector(qs, dataset.evidence, NltkAndPunctTokenizer().tokenize_paragraph_flat,
                               FastNormalizedAnswerDetector())


if __name__ == "__main__":
    main()