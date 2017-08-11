from collections import Counter
from typing import List, Optional, Iterable, Dict

import numpy as np
from nltk import PorterStemmer
from sklearn.feature_extraction.text import strip_accents_unicode, TfidfVectorizer
from sklearn.metrics import pairwise_distances

from configurable import Configurable
from data_processing.batching import get_samples
from data_processing.paragraph_qa import Document, SquadCorpus, DocumentQaStats
from dataset import Dataset, TrainingData
from squad.squad import clean_title
from utils import ResourceLoader, flatten_iterable, transpose_lists


class ParagraphQuestionGroup(object):
    def __init__(self, paragraphs: List[List[List[str]]],
                 questions: List[List[str]],
                 question_ids: List[str],
                 answer: np.ndarray,
                 group_id: Optional[object]):
        if len(questions) != len(answer):
            raise ValueError()
        self.question_ids = question_ids
        self.group_id = group_id
        self.paragraphs = paragraphs
        self.questions = questions
        self.answer = answer


class QuestionParagraphFeaturizer(Configurable):
    def init(self, train_corpus, resource_loader):
        pass

    def feature_names(self):
        raise NotImplementedError()

    def build_features(self, questions: List[List[str]],
                       paragraphs: List[List[List[str]]]) -> List[np.ndarray]:
        """
        @:return return [question_ix, word_ix, paragraph_ix, feature_ix] list of numpy arrays
        """
        raise NotImplementedError()


class PortStemmer(Configurable):
    def __init__(self):
        self.cache = {}
        self.stemmer = PorterStemmer()

    def normalize(self, x):
        stem = self.cache.get(x)
        if stem is None:
            stem = self.stemmer.stem(x)
            self.cache[x] = stem
        return stem

    def __call__(self, word):
        return self.normalize(word)

    def __setstate__(self, state):
        self.__init__()

    def __getstate__(self):
        return {}


class DocumentFeatureVectors(QuestionParagraphFeaturizer):
    def __init__(self, name: str, dist_fns: Iterable[str]=("cosine", ), normalizer=None,
                 stop_words=None, binary_question_fe: bool=False, other_params: Dict=None):
        self.stop_words = stop_words
        self.normalizer = normalizer
        self.other_params = other_params
        self.dist_fns = dist_fns
        self.binary_question_fe = binary_question_fe
        self._name = name

    def feature_names(self):
        fe = [self._name + "-" + fn for fn in self.dist_fns]
        if self.binary_question_fe:
            fe += ["binary-" + x for x in fe]
        return fe

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]) -> List[List[np.ndarray]]:
        docs = [" ".join(" ".join(s) for s in para) for para in paragraphs]
        params = dict(self.other_params) if self.other_params else {}
        if "stop" not in params and self.stop_words is not None:
            params["stop_words"] = self.stop_words.words
        params["tokenizer"] = lambda x: x.split(" ")
        if self.normalizer is not None:
            params["preprocessor"] = self.normalizer.normalize
        if "strip_accents" not in params:
            params["strip_accents"] = "unicode"

        vectorizer = TfidfVectorizer(**params)

        X = vectorizer.fit_transform(docs)

        # compute the distance for all question words at once
        all_question_words = flatten_iterable(questions)
        Q = vectorizer.transform(all_question_words)  # matrix of word x feature
        dists = [pairwise_distances(Q, X, fn) for fn in self.dist_fns]  # (dim, word, para)
        if self.binary_question_fe:
            Q = Q != 0
            dists += [pairwise_distances(Q, X, fn) for fn in self.dist_fns]
        dists = np.concatenate([np.expand_dims(x, 2) for x in dists])  # (word, para, dim)

        all_features = []
        ix = 0
        for q in questions:
            all_features.append(dists[ix:ix+len(q)])
            ix += len(q)

        return all_features


class WordMatchFeatures(QuestionParagraphFeaturizer):
    def __init__(self, normalizer, stop_words, any_match_features=False):
        self.normalizer = normalizer
        self.stop_words = stop_words
        self.any_match_features = any_match_features

    def feature_names(self):
        fe = ["exact_match", "lower_match", "stem_match"]
        if self.any_match_features:
            fe += [x + "_any" for x in fe]
        # fe += ["total_matched"]
        return fe

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]) -> List[np.ndarray]:
        norm_questions = [[strip_accents_unicode(w) for w in q] for q in questions]
        stop_words = self.stop_words.words
        n_features = len(self.feature_names())

        features_per_paragraph = []
        for paragraph in paragraphs:
            words = set()
            for sent in paragraph:
                words.update(sent)

            words = set(strip_accents_unicode(x) for x in words if x.lower() not in stop_words)
            lower_words = set(x.lower() for x in words)
            normalized_words = set(self.normalizer.normalize(word) for word in lower_words)

            features = []
            for question in norm_questions:
                question_features = np.zeros((len(question), n_features), dtype=np.float32)
                for i, word in enumerate(question):
                    if word in words:
                        question_features[i, :3] = 1
                    elif word.lower() in lower_words:
                        question_features[i, 1:3] = 1
                    elif self.normalizer.normalize(word) in normalized_words:
                        question_features[i, 2] = 1
                if self.any_match_features:
                    question_features[:, 3:6] = question_features[:, :3] > 0

                # question_features[:, -1] = (question_features[:, 2] > 0).sum()
                features.append(question_features)
            features_per_paragraph.append(features)

        # From (paragraph x question x (word x feature)) ->
        # (question x paragraph x (word x features))
        feature_per_question = transpose_lists(features_per_paragraph)

        for i, question_feature_list in enumerate(feature_per_question):
            # stack to get (paragraph x word x feature)
            # transpose to get (word x paragraph x feature)
            feature_per_question[i] = np.transpose(np.stack(question_feature_list), [1, 0, 2])
        return feature_per_question


def get_bigrams(sentence, stop):
    on_word = sentence[0]
    on_ix = 0
    for i, word in enumerate(sentence):
        if word.lower() in stop:
            continue
        yield (on_ix, (on_word, word))
        on_ix = i
        on_word = word


class BigramFeatures(QuestionParagraphFeaturizer):
    def __init__(self, normalizer, stop_words, skip_stop):
        self.normalizer = normalizer
        self.stop_words = stop_words
        self.skip_stop = skip_stop

    def feature_names(self):
        fe = ["bigram_match_count", "any_bigram_match"]
        return fe

    def build_features(self, questions: List[List[str]], paragraphs: List[List[List[str]]]) -> List[np.ndarray]:
        stop_words = self.stop_words.words
        bigram_stop = stop_words if self.skip_stop else set()

        norm_questions = [[strip_accents_unicode(w.lower()) for w in q] for q in questions]
        norm_questions = [[self.normalizer.normalize(w) for w in q] for q in norm_questions]

        features_per_paragraph = []
        for paragraph in paragraphs:
            bigrams = Counter()
            for sent in paragraph:
                sent = [strip_accents_unicode(x.lower()) for x in sent]
                words = list(self.normalizer.normalize(x) for x in sent)
                for _, bigram in get_bigrams(words, bigram_stop):
                    bigrams[bigram] += 1

            features = []
            for question in norm_questions:
                q_features = np.zeros((len(question), 2), dtype=np.float32)

                for i, bigram in get_bigrams(question, bigram_stop):
                    count = bigrams[(question[i], question[i+1])]
                    q_features[i, 0] = count
                    q_features[i, 1] = count > 0
                features.append(q_features)
            features_per_paragraph.append(features)

        # From (paragraph x question x (word x feature)) ->
        # (question x paragraph x (word x features))
        feature_per_question = transpose_lists(features_per_paragraph)

        for i, question_feature_list in enumerate(feature_per_question):
            # stack to get (paragraph x word x feature)
            # transpose to get (word x paragraph x feature)
            feature_per_question[i] = np.transpose(np.stack(question_feature_list), [1, 0, 2])
        return feature_per_question


def build_selection_group(doc: Document, max_questions, include_title) -> List[ParagraphQuestionGroup]:
    if max_questions is not None:
        raise NotImplementedError()

    questions = flatten_iterable([x.questions for x in doc.paragraphs])
    question_words = [x.words for x in questions]

    if include_title is not None:
        prefix = include_title
        title = clean_title(doc.title).split(" ")
        if prefix:
            question_words = [title + q for q in question_words]
        else:
            question_words = [q + title for q in question_words]

    paragraphs = [x.context for x in sorted(doc.paragraphs, key=lambda x: x.paragraph_num)]

    answers = np.concatenate([np.full(len(x.questions), x.paragraph_num, dtype=np.int32) for x in doc.paragraphs])
    doc_id = doc.title
    return [ParagraphQuestionGroup(paragraphs, question_words,
                                   [x.question_id for x in questions], answers, doc_id)]


class ParagraphSelectionTrainingData(TrainingData):
    """
    `ParagraphQuestionGroup` instances derived from each document in a `ParagraphQaCorpus` corpus
    """

    def __init__(self, corpus: SquadCorpus, max_question: None, include_title=None):
        self.max_question = max_question
        self.corpus = corpus
        self.include_title = include_title
        self._train = None
        self._test = None

    def _load_data(self):
        if self._train is not None:
            return
        print("Loading data for: " + self.corpus.name)
        self._train = self.corpus.get_train_docs()
        self._dev = self.corpus.get_dev_docs()

    def get_train_corpus(self) -> object:
        self._load_data()
        return DocumentQaStats(self._train)

    def get_resource_loader(self) -> ResourceLoader:
        return self.corpus.get_resource_loader()

    def get_train(self) -> Dataset:
        self._load_data()
        groups = flatten_iterable([build_selection_group(x, self.max_question, self.include_title) for x in self._train])
        return ParagraphSelectionDataset(self.corpus.name + "-train", groups)

    def get_eval(self) -> Dataset:
        self._load_data()
        groups = flatten_iterable([build_selection_group(x, self.max_question, self.include_title) for x in self._dev])
        return ParagraphSelectionDataset(self.corpus.name + "-eval", groups)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_train"] = None
        state["_dev"] = None
        return state

    def __setstate__(self, state):
        state["_train"] = None
        state["_dev"] = None
        self.__dict__ = state


class ParagraphSelectionDataset(Dataset):
    def __init__(self, name: str, points: List[ParagraphQuestionGroup]):
        self.name = name
        self.points = points

    def get_batches(self, n_epochs: int, n_elements: int = 0, is_train: bool = True):
        return get_samples(self.points, n_epochs, n_elements)

    def name(self):
        return self.name

    def __len__(self):
        return len(self.points)

    def get_voc(self):
        voc = set()
        for group in self.points:
            for question in group.questions:
                voc.update(question)
            for para in group.paragraphs:
                for sent in para:
                    voc.update(sent)
        return voc