from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import strip_accents_unicode
from tqdm import tqdm

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset
from docqa.utils import flatten_iterable


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    CYAN = "\033[96m"
    ENDC = '\033[0m'


def show_stats():
    splitter = MergeParagraphs(400)
    stop = NltkPlusStopWords(True)
    ranker = TopTfIdf(stop, 6)

    corpus = TriviaQaWebDataset()
    train = corpus.get_train()
    points = flatten_iterable([(q, d) for d in q.all_docs] for q in train)
    np.random.shuffle(points)

    counts = np.zeros(6)
    answers = np.zeros(6)
    n_answers = []

    points = points[:1000]
    for q, d in tqdm(points):
        doc = corpus.evidence.get_document(d.doc_id)
        doc = splitter.split_annotated(doc, d.answer_spans)
        ranked = ranker.prune(q.question, doc)
        counts[:len(ranked)] += 1
        for i, para in enumerate(ranked):
            if len(para.answer_spans) > 0:
                answers[i] += 1
        n_answers.append(tuple(i for i, x in enumerate(ranked) if len(x.answer_spans) > 0))

    print(answers/counts)
    c = Counter()
    other = 0
    for tup in n_answers:
        if len(tup) <= 2:
            c[tup] += 1
        else:
            other += 1

    for p in sorted(c.keys()):
        print(p, c.get(p)/len(points))
    print(other/len(points))


def show_web_paragraphs():
    splitter = MergeParagraphs(400)
    stop = NltkPlusStopWords(True)
    ranker = TopTfIdf(stop, 6)
    stop_words = stop.words

    corpus = TriviaQaWebDataset()
    train = corpus.get_train()
    points = flatten_iterable([(q, d) for d in q.all_docs] for q in train)
    np.random.shuffle(points)

    for q, d in points:
        q_words = {strip_accents_unicode(w.lower()) for w in q.question}
        q_words = {x for x in q_words if x not in stop_words}

        doc = corpus.evidence.get_document(d.doc_id)
        doc = splitter.split_annotated(doc, d.answer_spans)
        ranked = ranker.dists(q.question, doc)
        if len(ranked) < 2 or len(ranked[1][0].answer_spans) == 0:
            continue
        print(" ".join(q.question))
        print(q.answer.all_answers)
        for i, (para, dist) in enumerate(ranked[0:2]):
            text = flatten_iterable(para.text)
            print("Start=%d, Rank=%d, Dist=%.4f" % (para.start, i, dist))
            if len(para.answer_spans) == 0:
                continue
            for s, e in para.answer_spans:
                text[s] = bcolors.CYAN + text[s]
                text[e] = text[e] + bcolors.ENDC
            for i, w in enumerate(text):
                if strip_accents_unicode(w.lower()) in q_words:
                    text[i] = bcolors.ERROR + text[i] + bcolors.ENDC
            print(" ".join(text))
        input()


def show_open_paragraphs(start: int, end: int):
    splitter = MergeParagraphs(400)
    stop = NltkPlusStopWords(True)
    ranker = ShallowOpenWebRanker(6)
    stop_words = stop.words

    print("Loading train")
    corpus = TriviaQaOpenDataset()
    train = corpus.get_dev()
    np.random.shuffle(train)

    for q in train:
        q_words = {strip_accents_unicode(w.lower()) for w in q.question}
        q_words = {x for x in q_words if x not in stop_words}

        para = []
        for d in q.all_docs:
            doc = corpus.evidence.get_document(d.doc_id)
            para += splitter.split_annotated(doc, d.answer_spans)

        ranked = ranker.prune(q.question, para)
        if len(ranked) < start:
            continue
        ranked = ranked[start:end]

        print(" ".join(q.question))
        print(q.answer.all_answers)
        for i in range(start, end):
            para = ranked[i]
            text = flatten_iterable(para.text)
            print("Start=%d, Rank=%d" % (para.start, i))
            if len(para.answer_spans) == 0:
                # print("No Answer!")
                continue
            for s, e in para.answer_spans:
                text[s] = bcolors.CYAN + text[s]
                text[e] = text[e] + bcolors.ENDC
            for i, w in enumerate(text):
                if strip_accents_unicode(w.lower()) in q_words:
                    text[i] = bcolors.ERROR + text[i] + bcolors.ENDC
            print(" ".join(text))
        input()


if __name__ == "__main__":
    show_open_paragraphs(0, 4)