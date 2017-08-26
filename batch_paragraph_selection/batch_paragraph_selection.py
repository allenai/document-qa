import json
import unicodedata
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pandas import DataFrame
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from tqdm import tqdm

from data_processing.text_utils import WordNormalizer
from squad.squad_data import Document, SquadCorpus
from utils import flatten_iterable, ResourceLoader

"""
Somewhat defunct module for doing paragraph selection 
with many questions <-> to many paragraphs at once
"""


class ParagraphSelectorFeaturizer(object):

    def init(self, docs: List[Document], loader: ResourceLoader):
        pass

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        raise NotImplemented()

    def feature_names(self) -> List[str]:
        raise NotImplemented()


class AverageWordVecFeatures(ParagraphSelectorFeaturizer):
    def  __init__(self, vec, stop, question_feature: bool=False, context_features: bool=False,
                 dot_features: bool=False, sum_features: bool=False):
        self.stop = stop
        self.question_feature = question_feature
        self.context_features = context_features
        self.dot_features = dot_features
        self.sum_features = sum_features
        self.vecs = vec

    def feature_names(self):
        dim = len(next(iter(self.vecs.values())))
        fe = ["average_vec_dist"]
        if self.question_feature:
            fe += ["qvec%d" % i for i in range(dim)]
        if self.context_features:
            fe += ["cvec%d" % i for i in range(dim)]
        if self.dot_features:
            fe += ["dot%d" % i for i in range(dim)]
        if self.sum_features:
            fe += ["dot%d" % i for i in range(dim)]
        return fe

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        vec_dict = self.vecs
        dim = len(next(iter(vec_dict.values())))

        question_vectors = []
        for question in questions:
            vec = np.zeros(dim, dtype=np.float32)
            for word in question:
                lw = word.lower()
                if lw in self.stop:
                    continue
                word_vec = vec_dict.get(word)
                if word_vec is None:
                    word_vec = vec_dict.get(lw)
                if word_vec is not None:
                    vec += word_vec
            question_vectors.append(vec)

        context_vectors = []
        for para in paragraphs:
            vec = np.zeros(dim, dtype=np.float32)
            count = 0
            for word in flatten_iterable(para):
                lw = word.lower()
                if lw in self.stop:
                    continue
                word_vec = vec_dict.get(word)
                if word_vec is None:
                    word_vec = vec_dict.get(lw)
                if word_vec is not None:
                    count += 1
                    vec += word_vec
            context_vectors.append(vec/count)

        question_vectors = np.array(question_vectors)
        context_vectors = np.array(context_vectors)
        dists = pairwise_distances(question_vectors, context_vectors, "cosine").ravel()
        features = [np.expand_dims(1 - dists, 1)]
        if self.question_feature:
            features.append(question_vectors)
        if self.context_features:
            features.append(question_vectors)
        if self.dot_features:
            features.append(np.einsum("ak,bk->abk", question_vectors, context_vectors).reshape((-1, dim)))
        if self.sum_features:
            features.append(np.expand_dims(question_vectors, 0) +
                            np.expand_dims(context_vectors, 1)).reshape((-1, dim))

        return np.concatenate(features, axis=1)


class SentenceVecFeatures(ParagraphSelectorFeaturizer):
    def  __init__(self, vecs, stop):
        self.vecs = vecs
        self.stop = stop
        self.dim = len(next(iter(self.vecs.values())))

    def feature_names(self):
        return ["max_vec_dist"]

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        dim = self.dim

        question_vectors = []
        for question in questions:
            vec = np.zeros(dim, dtype=np.float32)
            for word in question:
                lw = word.lower()
                if lw in self.stop:
                    continue
                word_vec = self.vecs.get(word)
                if word_vec is None:
                    self.vecs.get(lw)
                if word_vec is not None:
                    vec += word_vec
            question_vectors.append(vec)

        context_vectors = []
        for para in paragraphs:
            for sent in para:
                vec = np.zeros(dim, dtype=np.float32)
                for word in sent:
                    lw = word.lower()
                    if lw in self.stop:
                        continue
                    word_vec = self.vecs.get(word)
                    if word_vec is not None:
                        vec += word_vec
                context_vectors.append(vec)

        question_vectors = np.array(question_vectors)
        context_vectors = np.array(context_vectors)
        dists = 1-pairwise_distances(question_vectors, context_vectors, "cosine").ravel()

        paragraph_features = []
        ix = 0
        for _ in questions:
            for para in paragraphs:
                paragraph_features.append(dists[ix:ix+len(para)].max())
                ix += len(para)

        return np.array(paragraph_features).reshape(-1, 1)


class WordMatchFeatures(ParagraphSelectorFeaturizer):
    def __init__(self, normalizer, stop, is_named=None, any_features=True):
        self.word_normalizer = normalizer
        self.is_named = is_named
        self.any_features = any_features
        self._word_features = {}
        self.stop = stop

    def feature_names(self):
        fe = ["match", "lower_match", "norm_match"]
        if self.any_features:
            fe += ["any_"+x for x  in fe]
        fe += ["num_question_words_matched", "percent_question_words_matched"]
        return fe

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        features = []
        for question in questions:
            question_words = set(word for word in question if word.lower() not in self.stop)
            question_words_lower = {word.lower() for word in question_words}
            question_words_normalized = {self.word_normalizer.normalize(word) for word in question_words_lower}

            for para in paragraphs:
                matched_words = set()

                sum_features = np.zeros(3)
                for word in flatten_iterable(para):
                    norm_word = self.word_normalizer.normalize(word)
                    matched = True
                    if word in question_words:
                        sum_features[:3] += 1
                        # if is_number(word):
                        #     sum_features[3] += 1
                        # if word[0].isupper():
                        #     sum_features[4] += 1
                        # if self.is_named is not None and self.is_named(word):
                        #     sum_features[5] += 1
                    elif word.lower in question_words_lower:  # lower match
                        sum_features[1:3] += 1
                    elif norm_word in question_words_normalized:
                        sum_features[2] += 1
                    else:
                        matched = False
                    if matched:
                        matched_words.add(norm_word)

                n_words = len(question_words_normalized)
                n_matched = len(matched_words)
                match_features = np.array([n_matched, n_matched/n_words])
                # match_features[0] = 0
                # match_features[:] = 0
                features.append(np.concatenate([sum_features, match_features], axis=0))

        features = np.array(features)
        if self.any_features:
            return np.concatenate([features[:, :-2], features[:, :-2] > 0, features[:, -2:]], axis=1)
        else:
            return features


class VectorizedDistanceFeatures(ParagraphSelectorFeaturizer):
    def __init__(self, vectorizer, name, distance_fn):
        self.name = name
        self.distance_fn = distance_fn
        self.vectorizer = vectorizer
        self.word_normalizer = WordNormalizer()

    def feature_names(self):
        return [self.name + "-" + x for x in self.distance_fn]

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        docs = [" ".join(" ".join(s) for s in para) for para in paragraphs]
        docs += [" ".join(q) for q in questions]
        X = self.vectorizer.fit_transform(docs)
        question_features = X[len(paragraphs):]
        doc_features = X[:len(paragraphs)]
        dists = [pairwise_distances(question_features, doc_features, x).ravel() for x in self.distance_fn]
        return 1 - np.concatenate([np.expand_dims(x, 1) for x in dists], axis=1)


class SentenceVectorizedDistanceFeatures(ParagraphSelectorFeaturizer):
    def __init__(self, vectorizer, name, distance_fn=("cosine",), fit_questions: bool=False):
        self.name = name
        self.distance_fn = distance_fn
        self.vectorizer = vectorizer
        self.word_normalizer = WordNormalizer()
        self.fit_questions = fit_questions

    def feature_names(self):
        return [self.name + "-" + x for x in self.distance_fn]

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]], ):
        docs = flatten_iterable([[" ".join(s) for s in para] for para in paragraphs])
        questions = [" ".join(q) for q in questions]
        if self.fit_questions:
            X = self.vectorizer.fit_transform(docs + questions)
            question_features = X[-len(questions):]
        else:
            X = self.vectorizer.fit_transform(docs)
            question_features = self.vectorizer.transform(questions)

        docs += [" ".join(q) for q in questions]
        paragraph_features = []
        ix = 0
        for para in paragraphs:
            paragraph_features.append(X[ix:ix + len(para)].max(axis=0))
            ix += len(para)
        paragraph_features = sparse.vstack(paragraph_features)

        dists = [pairwise_distances(question_features, paragraph_features, x).ravel() for x in self.distance_fn]
        return 1 - np.concatenate([np.expand_dims(x, 1) for x in dists], axis=1)


class ContextFeatures(ParagraphSelectorFeaturizer):
    def feature_names(self):
        return ["n_words", "n_sent"]

    def build_features(self, questions: List[List[List[str]]], paragraphs: List[List[List[str]]]):
        para_features = np.log(np.array([(sum(len(s) for s in x), len(x)) for x in paragraphs]))
        return np.repeat(para_features, len(questions), axis=0)



# ac = re.compile("[A-Z][A-Z\.?]+s?$")
# def extract_question_abrivations(words):
#     acc = []
#     for word in words:
#         if ac.match(word) is not None:
#             acc.append(word.replace(".", ""))
#
#     on = []
#     acc = []
#     for word in words[1:]:
#         if word[0].isupper() and word[1:].islower():
#             on.append(word[0])
#         elif len(on) > 0 and word not in ["of", "the"]:
#             acc.append(on)
#             on = []
#
#     if len(acc) > 1:
#         print(words)
#         print(" ".join("".join(s) for s in acc))


def hueristic(docs: List[Document]):
    stop = set(stopwords.words('english'))
    prune = set(stop)
    prune.update(["many", "how", "?", ",", "-", ".", "\"", "wa"])
    # prune.update(["many", "how", "?", ",", "do", "did", "does", "-", ".", "wa", "\"", "like",
    #               "also", "accord", "date", "mean", "area", "locat", "type", "color", "event",
    #               "number", "becom", "becam", "has])

    stemmer = PorterStemmer()
    normalize_cache = {}

    def normalize(x):
        lw = x.lower()
        norm = normalize_cache.get(lw)
        if norm is not None:
            return norm

        lw = unicodedata.normalize('NFKD', lw).encode('ascii', 'ignore').decode("ascii")
        stem = stemmer.stem(lw)
        normalize_cache[lw] = stem
        return stem

    total = 0
    errors = 0
    n_pruned = 0
    n_para = 0

    rng = np.random.RandomState(0)

    good_match = Counter()
    bad_match = Counter()

    for doc in docs:
        all_questions = flatten_iterable(x.questions for x in doc.paragraphs)
        rng.shuffle(all_questions)
        all_questions = all_questions[:100]
        context_bags = [Counter(normalize(x) for x in flatten_iterable(x.context)) for x in doc.paragraphs]
        # context_caps = [x in flatten_lists(flatten_lists(x.context)) for x in doc.paragraphs if x.is_upper()]
        for question in all_questions:
            total += 1

            for i, para in enumerate(doc.paragraphs):
                if any(x.question_id == question.question_id for x in para.questions):
                    correct = i
                    break

            n_para += len(doc.paragraphs)

            question_words = set(normalize(x) for x in question.words if x.lower())
            pruned_question_words = list(question_words - prune)
            question_caps = [x for x in flatten_iterable(question.words) if x.upper()]

            if any(x in question_words for x in ["it", "he", "she", "his", "her", "they", "this"]):
                # n_pruned += len(doc.paragraphs)
                pass
            else:
                candidates = []
                for i in range(len(doc.paragraphs)):
                    bag = context_bags[i]
                    matches = [w for w in pruned_question_words if w in bag]
                    if len(matches) > 0:
                        candidates.append(i)
                        if i == correct:
                            if np.random.random() < 0.05:
                                print(" ".join(question.words))
                                print(matches)
                            # if matches == ["ha"]:
                            #     print(" ".join(question.words))
                            #     print(matches)
                            if len(matches) == 1:
                                good_match.update(matches)
                        else:
                            # if np.random.random() < 0.005:
                            #     print(" ".join(question.words))
                            #     print(matches)
                            if len(matches) == 1:
                                bad_match.update(matches)


                n_pruned += len(doc.paragraphs) - len(candidates)
                if correct not in candidates:
                    errors += 1
                    # print()
                    # print(doc.wiki_title)
                    # print(" ".join(question.words))
                    # print(list(set(x.text for x in question.answer)))
                    # print(" ".join(" ".join(s) for s in doc.paragraphs[correct].context))
                    # input()

    print("Pruned %d/%d (%.4f)" % (n_pruned, n_para, n_pruned/n_para))
    print("Erors: %d/%d (%.4f)" % (errors, total, errors/total))


def build_features(docs: List[Document], resource_loader: ResourceLoader, n_sample_docs=None, n_sample_qs=None,
                   seed = None, dev=None):

    rng = np.random.RandomState(seed)

    stop = set(stopwords.words('english'))
    stop.update(["many", "how", "?", ",", "-", "."])

    # normalizer_up = WordNormalizer()
    normalizer_lw = WordNormalizer()
    # vecs = resource_loader.load_word_vec("glove.6B.100d")

    featurizers = [
        VectorizedDistanceFeatures(TfidfVectorizer(tokenizer=lambda x: x.split(" "), stop_words=stop,
                                                   lowercase=False, preprocessor=normalizer_lw.normalize),
                                   "doc-tfidf", ["cosine"]),
        # VectorizedDistanceFeatures(TfidfVectorizer(tokenizer=lambda x: x.split(" "), stop_words=stop,
        #                                            ngram_range=(2, 2),
        #                                            lowercase=False, preprocessor=normalizer_lw.normalize),
        #                            "doc-tfidf-bigrams", ["cosine"]),
        # SentenceVectorizedDistanceFeatures(TfidfVectorizer(tokenizer=lambda x: x.split(" "), stop_words=stop,
        #                                                    preprocessor=normalizer_lw.normalize,
        #                                                    lowercase=False), "tfidf-sentence", ["cosine"],
        #                                    fit_questions=False),
        # SentenceVectorizedDistanceFeatures(TfidfVectorizer(tokenizer=lambda x: x.split(" "), stop_words=stop,
        #                                                    ngram_range=(1, 2),
        #                                                    preprocessor=normalizer_lw.normalize,
        #                                                    lowercase=False), "tfidf-sentence-bigram", ["cosine"]),
        WordMatchFeatures(normalizer_lw, stop, None, True),
        # SentenceVecFeatures(vecs, stop),
        # AverageWordVecFeatures(resource_loader.load_word_vec("glove.6B.100d"), stop)
        # ContextFeatures(),
    ]

    if n_sample_docs is not None:
        rng.shuffle(docs)
        docs = docs[:n_sample_docs]

    n_train = len(docs)
    if dev is not None:
        docs += dev

    feature_names = ["label"] + flatten_iterable(x.feature_names() for x in featurizers)
    built_features = []
    other_features = []
    index = []

    for i, doc in enumerate(tqdm(docs)):
        is_train = i < n_train
        all_questions = flatten_iterable(x.questions for x in doc.paragraphs)
        if is_train and n_sample_qs is not None:
            rng.shuffle(all_questions)
            all_questions = all_questions[:n_sample_qs]

        labels = []
        for question in all_questions:
            for para in doc.paragraphs:
                other_features.append([doc.title, "train" if is_train else "dev"])
                index.append((question.question_id, para.paragraph_num))
                if any(x.question_id == question.question_id for x in para.questions):
                    labels.append(1)
                else:
                    labels.append(0)

        para_text = [x.context for x in doc.paragraphs]
        question_text = [x.words for x in all_questions]

        features = np.concatenate([np.array(labels).reshape(-1, 1)] +
                                  [fe.build_features(question_text, para_text) for fe in featurizers], axis=1)
        if features.shape[1] != len(feature_names):
            raise ValueError(features.shape[1], len(feature_names))
        built_features.append(features)

    built_features = np.concatenate(built_features, axis=0)

    features = {}
    for i in range(len(feature_names)):
        features[feature_names[i]] = built_features[:, i]

    features["article_title"] = [x[0] for x in other_features]
    features["source"] = [x[1] for x in other_features]

    return DataFrame(features, index=pd.MultiIndex.from_tuples(index, names=['question_id', 'paragraph_num']))
    # with open(output, "w") as f:
    #     f.write("\t".join(feature_names + ["article_title", "question_id"]))
    #     f.write("\n")
    #     for i, arr in enumerate(all_features):
    #         f.write("\t".join(str(x) for x in arr))
    #         f.write("\t")
    #         f.write("\t".join(text_features[i]))
    #         f.write("\n")


def get_classifier_cv_scores(df, n_folds=5):
    features = [c for c in df.columns if c not in {"label"} and df[c].dtype != np.object]
    X = df[features]
    y = df.label

    clf = LogisticRegression()
    # clf = SVC(probability=True)
    # clf = RandomForestClassifier(100)

    all_articles = list(set(df.article_title))
    np.random.shuffle(all_articles)
    all_articles = np.array(all_articles)
    test_pred = np.full(len(y), np.nan)

    for train, test in KFold(n_folds).split(all_articles):
        train_ix = df.article_title.isin(set(all_articles[train]))
        test_ix = df.article_title.isin(set(all_articles[test]))
        clf.fit(X[train_ix], y[train_ix])
        test_pred[test_ix] = clf.predict_proba(X[test_ix])[:, 1]

    df["clf_cv_predictions"] = test_pred


def get_classifier_dev_scores(df):
    features = [c for c in df.columns if c not in {"label"} and df[c].dtype != np.object]
    X = df[df.source == "train"][features]
    y = df[df.source == "train"].label
    clf = LogisticRegression()
    clf.fit(X, y)
    df["clf_train_predictions"] = clf.predict_proba(df[features])[:, 1]


def convert_to_rank(features):
    for key, question_df in df.groupby(level="question_id"):
        correct = question_df.label.argmax()
        ranks = np.array([rankdata(question_df[feature])[correct] for feature in features])


def compute_ranks_features(df, features):
    feature_ranks = []
    n_para = []

    # for key, question_df in df.groupby(level="question_id"):
    #     feature_ranks.append(np.array([rankdata(-question_df[feature])[correct] for feature in features]))
    #     n_para.append(len(question_df))
    #     question_df

    # n_para = np.array(n_para)
    # feature_ranks = np.array(feature_ranks)
    #
    # for i, feature in enumerate(features):
    #     ranks = feature_ranks[:, i]
    #     rank_per = feature_ranks[:, i]/n_para
    #     print(feature)
    #     print("Abs Rank: %.4f, %.4f" % (ranks.mean(), rank_per.mean()))
    #     print("Top1:  %.4f" % (ranks < 2).mean())
    #     print("Top3:  %.4f" % (ranks < 4).mean())
    #     print("Top5:  %.4f" % (ranks < 6).mean())
    #     print("Top10: %.4f" % (ranks < 11).mean())


def highlight_words(para: List[List[str]], words):
    COLOR = '\033[94m'
    ENDC = '\033[0m'

    highlights = []
    for sent in para:
        highlights.append(" ".join((COLOR + word + ENDC if word.lower() in words else word) for word in sent))
    return " ".join(highlights)


def show_errors(df, docs: List[Document], feature_name):
    stop = set(stopwords.words('english'))
    stop.update(["many", "how", "?", ",", "-", ".", "\"", "wa"])


    id_to_para = {}
    id_to_question = {}
    for doc in docs:
        for para in doc.paragraphs:
            for question in para.questions:
                id_to_para[question.question_id] = doc
                id_to_question[question.question_id] = question

    for key, question_df in df.groupby(level="question_id"):
        correct = question_df.label.argmax()
        feature = question_df[feature_name]
        ranks = feature.ranked_questions(ascending=False)
        rank = ranks[correct]
        if rank > 3:
            question = id_to_question[key].words
            question_words = set(x.lower() for x in question if x.lower() not in stop)
            r_sorted = ranks.sort_values()
            pred = r_sorted.index[0]
            pred2 = r_sorted.index[1]

            doc = id_to_para[key]
            print()
            print("Rank=%d, article=%s" % (rank, id_to_para[key]))
            print(highlight_words([question], question_words))
            print("*" * 10 + " CORRECT (predicted %d, %.4f) " % (rank, feature[correct]) + "*"*10)
            print(highlight_words(doc.paragraphs[correct[1]].context, question_words))

            print("*" * 10 + " ERROR1 (predicted %d, %.4f) " % (ranks[pred], feature[pred]) + "*" * 10)
            print(highlight_words(doc.paragraphs[pred[1]].context, question_words))

            print("*" * 10 + " ERROR2 (predicted %d, %.4f) " % (ranks[pred2], feature[pred2]) + "*" * 10)
            print(highlight_words(doc.paragraphs[pred2[1]].context, question_words))

            # print("*" * 10 + " PREDICTED " + "*" * 10)
            # predicted = question_df[feature]


def show_eval(df, features=None):
    if features is None:
        features = [f for f in df.columns if f != "label" and df[f].dtype != np.object]

    print("Computing ranks...")
    rank_df = df.groupby(level="question_id")[features].ranked_questions(ascending=False)

    print("Eval")
    for f in rank_df.columns:
        ranks = rank_df[f][df.label == 1]
        print(f)
        print("Abs Rank: %.4f" % (ranks.mean()))
        print("Top1:  %.4f" % (ranks < 2).mean())
        print("Top3:  %.4f" % (ranks < 4).mean())
        print("Top5:  %.4f" % (ranks < 6).mean())
        print("Top10: %.4f" % (ranks < 11).mean())


def save_prediction(df, feature, output):
    answer_dict = {}
    for question_id, question_df in df.groupby(level="question_id"):
        values = question_df[feature].sort_index()
        answer_dict[question_id] = list(values.values)

    with open(output, "w") as f:
        json.dump(answer_dict, f)

if __name__ == "__main__":
    corp = SquadCorpus()
    print("Loading...")
    docs = corp.get_train()[:5]
    print("Building features....")
    df = build_features(docs, corp.get_resource_loader(), None, None, seed=0, dev=corp.get_dev()[:5])

    print("Classifier..,")
    get_classifier_dev_scores(df)

    show_eval(df[df.source == "dev"])

    # print("Saving...")
    # save_prediction(df[df.source == "dev"], "clf_train_predictions", "/tmp/paragraph_pred.json")


    # show_errors(df, docs, "clf_predictions")

    # show_errors(df, )
        # df.concat(df.groupby(level="question_id")[features].rank(), copy=False)
    # print(df.columns)
    # # fe = ["match", "lower_match", "norm_match", "clf_predictions", "tfidf-cosine", "average_vec_dist"]
    # # fe = ["tfidf-cosine", "tfidf-norm-cosine"]
    # fe = [x for x in df.columns if x not in {"label", "paragraph_num"} and df[x].dtype != np.object and not x.startswith("vec")]
    # eval_features(df, docs, fe)
