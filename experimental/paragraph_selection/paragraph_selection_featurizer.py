import re

import numpy as np
from typing import List
from collections import defaultdict, Counter

from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from configurable import Configurable
from data_processing.document_splitter import ExtractedParagraphWithAnswers
from data_processing.text_utils import NltkPlusStopWords, WordNormalizer
from data_processing.word_vectors import load_word_vectors
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from utils import flatten_iterable, print_table, split, group
import pickle



class ParagraphSelectionFeaturizer(object):
    def get_feature_names(self) -> List[str]:
        raise NotImplementedError()

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]) -> np.ndarray:
        raise NotImplementedError()


class JointParagraphSelectionFeaturizer(Configurable):

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError()

    def get_word_feature_names(self) -> List[str]:
        raise NotImplementedError()

    def get_joint_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        raise NotImplementedError()


class ParagraphOrderFeatures(ParagraphSelectionFeaturizer):
    def get_feature_names(self) -> List[str]:
        return ["first", "word_start", "log_word_start", "inv_word_start"]

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        paragraphs = flatten_iterable(paragraphs)
        starts = np.array([x.start for x in paragraphs], dtype=np.float32)
        starts /= 400  # Keep the scale a bit more reasonable
        return np.stack((starts == 0, starts, np.log(starts+1), 1/(starts + 1)), axis=1)


class ParagraphFeatures(ParagraphSelectionFeaturizer):
    def get_feature_names(self):
        return ["len", "log_len", "inv_len"]

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        paragraphs = flatten_iterable(paragraphs)
        arr = np.array([(x.end-x.start) for x in paragraphs], dtype=np.float32)
        arr /= 400
        return np.stack([arr, np.log(arr+1), 1/(arr+1)], axis=1)


class LocalTfIdfFeatures(ParagraphSelectionFeaturizer):
    def __init__(self, stop, use_question_voc=False,
                 per_document=True, normalization=None, name=None):
        self.stop = stop
        self.normalization = normalization
        self.use_question_voc = use_question_voc
        self.name = name
        self.per_document = per_document

    def get_feature_names(self):
        return ["local-tfidf-dist" if self.name is None else self.name]

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        if self.use_question_voc:
            target_words = set()
            for word in question:
                word = word.lower()
                if word not in self.stop.words and re.match("\w\w+", word):
                    target_words.add(word)
        else:
            target_words = None

        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words,
                                vocabulary=target_words,
                                ngram_range=(1,1), token_pattern=r"(?u)\b\w\w+\b")
        if not self.per_document:
            paragraphs = [flatten_iterable(paragraphs)]

        out = []
        for doc in paragraphs:
            text = []
            for para in doc:
                if self.normalization:
                    text.append(" ".join(" ".join(self.normalization.normalize(w) for w in s) for s in para.text))
                else:
                    text.append(" ".join(" ".join(s) for s in para.text))
            try:
                para_features = tfidf.fit_transform(text)
            except ValueError:
                out.append(np.zeros((len(doc), 1), dtype=np.float32))
                continue
            q_features = tfidf.transform([" ".join(question)])
            dists = pairwise_distances(q_features, para_features, "cosine").ravel().astype(np.float32)
            out.append(dists.reshape((len(doc), 1)))
        return np.concatenate(out, axis=0)

#
# class WordMatchingNeighborFeaturizer(JointParagraphSelectionFeaturizer):
#     def __init__(self, stop_words, normalizer, word_vec: str, n_neighbors: int):
#         self.stop_words = stop_words
#         self.normalizer = normalizer
#         self.n_neighbors = n_neighbors
#         self.word_vecs = word_vec
#         self._vec = None
#         self._words = None
#         self._knn = None
#
#     def init(self):
#         if self._vec is None:
#             print("Loading...")
#             self._vec = load_word_vectors(self.word_vecs)
#             self._words = np.array(list(self._vec.keys()))
#             print("Building nn")
#             self._knn = NearestNeighbors(n_neighbors=50, algorithm="auto")
#             self._knn.fit(np.stack(list(self._vec.values()), axis=0))
#             print("Done")
#
#     def clear(self):
#         self._vec = None
#         self._words = None
#         self._knn = None
#     def get_feature_names(self):
#         return []
#
#     def get_word_feature_names(self):
#         return ["word_match", "word_lower_match", "word_normalized_match"]
#
#     def get_joint_features(self, question: List[str], paragraphs: List[AnnotatedParagraph]):
#         stop = self.stop_words.words
#
#         q_vecs = {}
#         for word in question:
#             if word in q_vecs:
#                 continue
#             lw = word.lower()
#             if lw in stop:
#                 continue
#             vec = self._vec.get(word)
#             if vec is None:
#                 word = word.lower()
#                 vec = self._vec.get(word)
#             if vec is not None:
#                 q_vecs[word] = vec
#
#         query_words = list(q_vecs.keys())
#         query = np.stack(list(q_vecs.values()), axis=0)
#         nns = self._knn.kneighbors(query, n_neighbors=self.n_neighbors, return_distance=False)
#         words_nns = {query_words[i]: self._words[ix] for i, ix in enumerate(nns)}
#
#         exact_map = defaultdict(list)
#         lw_map = defaultdict(list)
#         norm_map = defaultdict(list)
#         nn_map = defaultdict(list)
#
#         for i, word in enumerate(question):
#             lw = word.lower()
#             if lw in stop:
#                 continue
#             exact_map[word].append(i)
#             lw_map[lw].append(i)
#             norm_map[self.normalizer.normalize(word)].append(i)
#             nns = words_nns.get(word)
#             if nns is None:
#                 nns = words_nns.get(lw)
#             if nns is not None:
#                 for word in nns:
#                     nn_map[word].append(i)
#
#         features = np.zeros((len(paragraphs), len(question), 4), dtype=np.float32)
#         for i, para in enumerate(paragraphs):
#             q_feautres = features[i]
#             for sent in para.text:
#                 for word in sent:
#                     fe = exact_map.get(word)
#                     if fe is not None:
#                         q_feautres[fe, 0] += 1
#                         continue
#
#                     fe = lw_map.get(word.lower())
#                     if fe is not None:
#                         q_feautres[fe, 1] += 1
#                         continue
#
#                     fe = norm_map.get(self.normalizer.normalize(word))
#                     if fe is not None:
#                         q_feautres[fe, 2] += 1
#                         continue
#
#                     fe = nn_map.get(self.normalizer.normalize(word))
#                     if fe is not None:
#                         q_feautres[fe, 3] += 1
#
#         return features, None

class NGramMatchingFeaturizer(JointParagraphSelectionFeaturizer):
    def __init__(self, stop_words, normalizer, ngram_len=(1, 1)):
        self.stop_words = stop_words
        self.ngram_len = ngram_len
        self.normalizer = normalizer

    def get_word_feature_names(self):
        names = []
        for i in range(self.ngram_len[0], self.ngram_len[1]+1):
            i += 1
            names += ["%d_%s" % (i, x) for x in ["word_match", "word_lower_match", "word_normalized_match"]]
        return names

    def get_feature_names(self):
        return []

    def get_joint_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        paragraphs = flatten_iterable(paragraphs)
        ngram_s = self.ngram_len[0]
        ngram_e = self.ngram_len[1] + 1
        if self.stop_words is None:
            stop = {}
        else:
            stop = self.stop_words.words

        exact_map = defaultdict(list)
        lw_map = defaultdict(list)
        norm_map = defaultdict(list)

        lw = [x.lower() for x in question]
        norm = [self.normalizer.normalize(x) for x in question]
        for l in range(ngram_s, ngram_e):
            for i in range(0, len(question)-l+1):
                if lw[i] in stop:
                    continue
                e = i+l
                exact_map[tuple(question[i:e])].append(i)
                lw_map[tuple(lw[i:e])].append(i)
                norm_map[tuple(norm[i:e])].append(i)

        features = np.zeros((len(paragraphs), len(question), (ngram_e-ngram_s)*3), dtype=np.float32)

        for i, para in enumerate(paragraphs):
            q_feautres = features[i]
            para = flatten_iterable(para.text)
            lw = [x.lower() for x in para]
            norm = [self.normalizer.normalize(x) for x in para]
            ix = 0
            for l in range(ngram_s, ngram_e):
                for i in range(len(para) - l + 1):
                    e = i + l
                    fe = exact_map.get(tuple(para[i:e]))
                    if fe is not None:
                        q_feautres[fe, ix] += 1
                        continue

                    fe = lw_map.get(tuple(lw[i:e]))
                    if fe is not None:
                        q_feautres[fe, ix+1] += 1
                        continue

                    fe = norm_map.get(tuple(norm[i:e]))
                    if fe is not None:
                        q_feautres[fe, ix+2] += 1
                ix += 3
        return features, None


class WordMatchingFeaturizer(ParagraphSelectionFeaturizer):
    def __init__(self, fe: NGramMatchingFeaturizer):
        self.fe = fe

    def get_feature_names(self) -> List[str]:
        names = self.fe.get_word_feature_names()
        return ["sum_" + x for x in names] + ["any_" + x for x in names] + ["mean_" + x for x in names]

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]) -> np.ndarray:
        joint_featres = self.fe.get_joint_features(question, paragraphs)[0]
        word_sums = joint_featres[:, :, :].sum(axis=1)
        any_word = (joint_featres[:, :, :] > 0).sum(axis=1)
        mean_words = joint_featres[:, :, :].mean(axis=1)
        return np.concatenate([word_sums, any_word, mean_words], axis=1)


def tf_idf(evidence: TriviaQaEvidenceCorpusTxt, n_sample: int):
    output_filename = "/tmp/tfidf.pkl"
    if exists(output_filename):
        print("Loading tfidf from cache")
        with open(output_filename, "rb") as f:
            return pickle.load(f)
    docs = evidence.list_documents()
    np.random.shuffle(docs)
    docs = docs[:n_sample]
    stop = NltkPlusStopWords(punctuation=True).words
    tfidf = TfidfVectorizer(min_df=3, max_df=0.6,
                            strip_accents="unicode", stop_words=stop)
    all_docs = []
    for doc in tqdm(docs, "Loading Background Docs"):
        text = evidence.get_document(doc, flat=True)
        if text is None:
            raise ValueError()
        all_docs.append(" ".join(text))

    print("Fitting...")
    tfidf.fit(all_docs)

    print("Saving...")
    print(tfidf)
    with open(output_filename, "wb") as f:
        pickle.dump(tfidf, f)
    return tfidf
