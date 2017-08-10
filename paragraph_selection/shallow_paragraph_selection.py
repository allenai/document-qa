from collections import defaultdict
from typing import List
import pandas as pd
import pickle
import numpy as np
from os.path import exists

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

from data_analysis.analyze_full_document_eval import compute_model_scores
from data_processing.document_splitter import Truncate, MergeParagraphs, TopTfIdf, ExtractedParagraph, DocumentSplitter
from data_processing.text_utils import NltkPlusStopWords, WordNormalizer
from paragraph_selection.paragraph_selection_featurizer import LocalTfIdfFeatures, ParagraphOrderFeatures, \
    NusedWordFeatures, NGramMatchingFeaturizer, ParagraphFeatures
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import TriviaQaQuestion
from utils import flatten_iterable, print_table, split, partition


def build_features(questions: List[TriviaQaQuestion], splitter: DocumentSplitter,
                   corpus: TriviaQaEvidenceCorpusTxt, featurizers, log_every=None):

    all_feautres = []
    para_ranges = []
    doc_ids = []
    q_uids = []
    n_answers = []
    for i, question in enumerate(questions):
        if log_every is not None and i % log_every == 0:
            print("On question %d" % i)
        paragraphs = []
        for doc in question.all_docs:
            text = corpus.get_document(doc.doc_id)
            split = splitter.split(text, doc.answer_spans)
            paragraphs.extend(split)
            doc_ids += [doc.doc_id for _ in range(len(split))]

        paragraph_features = []
        for fe in featurizers:
            para_fe = fe.get_features(question.question, paragraphs)
            paragraph_features.append(para_fe)

        n_answers += [len(x.answer_spans) for x in paragraphs]
        q_uids += [question.question_id for _ in range(len(paragraphs))]
        q_features = np.concatenate(paragraph_features, axis=1)
        # q_features -= q_features.mean(axis=0)
        all_feautres.append(q_features)
        para_ranges.append(np.array([(x.start, x.end) for x in paragraphs]))

    feature_names = flatten_iterable(x.get_feature_names() for x in featurizers)
    all_feautres = np.concatenate(all_feautres)
    para_ranges = np.concatenate(para_ranges, axis=0)
    data = dict(quid=q_uids, doc_id=doc_ids, n_answers=n_answers,
                para_start=para_ranges[:, 0], para_end=para_ranges[:, 1])
    data.update({feature_names[i]: all_feautres[:, i] for i in range(len(feature_names))})
    df = pd.DataFrame(data)
    return df


def build_features_par(questions: List[TriviaQaQuestion], splitter: DocumentSplitter,
                   corpus: TriviaQaEvidenceCorpusTxt, featurizers, n_processes=2, chunk_size=100):
    if n_processes == 1:
        return build_features(tqdm(questions), splitter, corpus, featurizers)
    else:
        from multiprocessing import Pool
        chunks = split(questions, n_processes)
        chunks = flatten_iterable([partition(c, chunk_size) for c in chunks])
        print("Processing %d chunks with %d processes" % (len(chunks), n_processes))
        pool = Pool(n_processes)
        pbar = tqdm(total=len(questions))

        def call_back(results):
            pbar.update(len(set(results.quid)))

        results = [pool.apply_async(build_features,
                                    [c, splitter, corpus, featurizers], callback=call_back)
                   for c in chunks]
        results = [r.get() for r in results]
        pool.close()
        pool.join()
        return pd.concat(results, axis=0, ignore_index=True)


def get_classifier_dev_scores(df, features):
    X = df[df.source == "train"][features]
    y = df[df.source == "train"].label
    clf = LogisticRegression()
    clf.fit(X, y)
    print(clf.coef_.shape)
    for name, val in zip(features, clf.coef_[0]):
        print("%s: %.4f" % (name, val))
    df["clf_train_predictions"] = clf.predict_proba(df[features])[:, 1]


def get_classifier_cv_scores(df, features, clf, n_splits=2):
    X = df[features].values
    y = df.label.values
    quids = list(set(df["quid"]))
    quids.sort()
    quids = np.array(quids)
    predictions = np.zeros(len(df))

    for train_ix, test_ix in KFold(n_splits).split(quids):
        train_quid = set(quids[train_ix])
        test_quid = set(quids[test_ix])
        test_ix = [x in test_quid for x in df.quid]
        train_ix = [x in train_quid for x in df.quid]
        clf.fit(X[train_ix], y[train_ix])
        predictions[test_ix] = clf.predict_proba(X[test_ix])[:, 1]

    df["cv_predictions"] = predictions


def get_classifier_train_scores(df, features, clf):
    clf.fit(df[features].values, df.label.values)
    df["train_predictions"] = clf.predict_proba(df[features].values)[:, 1]
    # df["train_predictions"] = clf.predict(df[features].values)


class SplitterStats(object):
    def __init__(self):
        self.n_answers = 0
        self.per_answers = 0
        self.upper_bound = 0
        self.paragraphs = 0
        self.n_tokens = 0
        self.max_tokens = 0

    def update(self, paragraph: List[ExtractedParagraph]):
        total = sum(len(x.answer_spans) for x in paragraph)
        n_tokens = sum(sum(len(s) for s in x.text) for x in paragraph)
        self.n_answers += total
        self.per_answers += sum(len(x.answer_spans)>0 for x in paragraph) / len(paragraph)
        self.max_tokens = max(self.max_tokens, n_tokens)
        self.upper_bound += total > 0
        self.paragraphs += len(paragraph)
        self.n_tokens += n_tokens


def check_upper_bounds():
    stop = NltkPlusStopWords()
    norm = WordNormalizer()
    dataset = TriviaQaWebDataset()
    print("Load questions")
    questions = dataset.get_dev()
    questions = np.random.RandomState(5).choice(questions, 400, replace=False)
    splitters = {
                 "tf-idf": (MergeParagraphs(400, pad=False), TopTfIdf(stop, 10)),
                 "top": (MergeParagraphs(400, pad=False), None),
                 # "tf-idf-400": (MergeParagraphs(400, pad=False), TopTfIdf(stop, 10)),
                 # "tf-idf-2-400": (MergeParagraphs(400, pad=True), TopTfIdf(stop, 10))
                 }
    stats = defaultdict(SplitterStats)
    total = 0
    for q in tqdm(questions):
        for doc in q.all_docs:
            total += 1
            text = dataset.evidence.get_document(doc.doc_id)
            flattened = flatten_iterable(text)
            stats["all"].update([ExtractedParagraph(flattened, 0, sum(len(x) for x in flattened),
                                                   doc.answer_spans)])
            for name, (splitter, para_filter) in splitters.items():
                para = splitter.split(text, doc.answer_spans)
                if para_filter is not None:
                    para = para_filter.prune(q.question, para)
                for i in range(1, 5):
                    stats[name + "-%d" % i].update(para[:i])

    table = [["name", "upper_bound", "per_answers", "n_tokens", "paragraphs", "n_answers"]]
    for name, stat in sorted(stats.items(), key=lambda x:x[0]):
        table.append([name] + ["%.4f" % (x/total) for x in [stat.upper_bound, stat.per_answers,
                                                            stat.n_tokens, stat.paragraphs, stat.n_answers]])
    print_table(table)



def train_oracle():
    stop = NltkPlusStopWords()
    norm = WordNormalizer()

    if exists("/tmp/cache.pkl"):
        print("Loading from cache...")
        with open("/tmp/cache.pkl", "rb") as f:
            existing = pickle.load(f)
    else:
        existing = None

    featurizers = [LocalTfIdfFeatures(stop),
                   ParagraphOrderFeatures(),
                   ParagraphFeatures(),
                   NusedWordFeatures(NGramMatchingFeaturizer(None, norm, (1, 1)))]
    # featurizers = []

    dataset = TriviaQaWebDataset()
    index_features = ["quid", "doc_id", "para_start", "para_end"]

    if existing is None:
        print("Load questions")
        questions = dataset.get_train()
        questions = np.random.RandomState(5).choice(questions, 1000, replace=False)

        print("Build features")
        df = build_features_par(questions, MergeParagraphs(400),
                        dataset.evidence, featurizers, n_processes=2, chunk_size=500)

        if existing is not None:
            df = df.merge(existing, on=index_features, copy=False)
        else:
            print("Caching...")
            with open("/tmp/cache.pkl", "wb") as f:
                pickle.dump(df, f)
    else:
        df = existing

    features = [x for x in df.columns if (x not in index_features and x != "n_answers")]
    # features = ["local-tfidf-dist"]
    df["label"] = df["n_answers"] > 0

    print("Classifying")
    print(features)
    clf = LogisticRegression(C=10000)
    # get_classifier_cv_scores(df, features, clf, 5)
    get_classifier_train_scores(df, features, clf)
    print(clf.coef_.shape)
    for name, val in sorted(zip(features, clf.coef_[0]), key=lambda x:abs(x[1])):
        print("%s: %.4f" % (name, val))

    # print((df["label"] == (df["train_predictions"] > 0.5)).mean())

    scores = {}
    score_features = [("local-tfidf-dist", True), ("para_start", True), ("train_predictions", False)]
    for fe, asc in score_features:
        clf.fit(df[[fe]].values, df.label)
        pred = clf.predict_proba(df[[fe]].values)[:, 1]
        print(fe)
        print((df["label"] == (pred > 0.5)).mean())

        df.sort_values(["quid", fe], inplace=True, ascending=asc)
        clf_scores = compute_model_scores(df, target_score="label", max_over="label", by_doc=True)
        scores[fe] = clf_scores

    cols = list(scores.keys())
    rows = [["Rank"] + cols]
    for i in range(10):
        rows.append(["%d" % (i+1)] + ["%.4f" % scores[k][i] for k in cols])
    print_table(rows)


def label_clf():
    stop = NltkPlusStopWords()
    norm = WordNormalizer()
    dataset = TriviaQaWebDataset()

    print("Load questions")
    questions = dataset.get_dev() + dataset.get_train()

    print("Load labels")
    dev = pd.read_csv("/Users/chrisc/Desktop/open-dev-eval-400.csv")
    dev = pd.read_csv("/Users/chrisc/Desktop/bidaf-sum-answers/dev-merge-800.csv")
    dev["source"] = "dev"

    train = pd.read_csv("/Users/chrisc/Desktop/bidaf-sum-answers/train-merge-800-3k.csv")
    train["source"] = "train"

    rng = np.random.RandomState(0)
    quids = set(rng.choice(list(set(dev.quid)), 2000, replace=False))
    quids.update(rng.choice(list(set(train.quid)), 0, replace=False))

    labels = pd.concat([train, dev], axis=0, ignore_index=True)
    labels = labels[[x in quids for x in labels.quid]]
    questions = [q for q in questions if q.question_id in quids]
    del train, dev

    print("Build features")
    fe = build_features_par(questions, MergeParagraphs(400),
                        dataset.evidence,
                        [
                                # TfIdfFeatures(tf_idf(dataset.evidence, 10000)),
                                # ParagraphOrderFeatures(),
                                # FastLocalTfIdfFeatures(stop, None),
                                LocalTfIdfFeatures(stop),
                            # SumWordFeatures(TfIdfFeatures(stop, norm)),
                            #     NusedWordFeatures(WordMatchingFeaturizer(stop, norm))
                         ], n_processes=2)

    index_features = ["quid", "doc_id", "para_start", "para_end"]
    df = labels.merge(fe, on=index_features, copy=False)
    features = [x for x in fe.columns if x not in index_features]

    df["label"] = df["text_f1"] > 0.9

    print("Classifying")
    get_classifier_dev_scores(df, features)

    #
    # df = df[df.source == "dev"]
    # acc = ((df.clf_train_predictions > 0.5) == (df.label == 1)).mean()
    # print("Accuracy = %.4f" % acc)

    scores = {}
    features = [("clf_train_predictions", False), ("local-tfidf-dist", True)]
    # features = [("local-tfidf-dist", True)]
    for fe, asc in features:
        print(fe)
        df.sort_values(["quid", fe], inplace=True, ascending=asc)
        clf_scores = compute_model_scores(df, by_doc=True)
        scores[fe] = clf_scores

    df.sort_values(["quid", "para_start", "para_end"], inplace=True)
    scores["FirstN"] = compute_model_scores(df, by_doc=True)

    cols = list(scores.keys())
    rows = [["Rank"] + cols]
    for i in range(10):
        rows.append(["%d" % (i+1)] + ["%.4f" % scores[k][i] for k in cols])
    print_table(rows)


if __name__ == "__main__":
    train_oracle()
