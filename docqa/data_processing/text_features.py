import re

import numpy as np
from nltk import Counter, WordNetLemmatizer

from docqa.configurable import Configurable

"""
Adding classic/shallow text features, I have only done shallow experiments with these 
and not found them to be of much use 
"""


any_num_regex = re.compile("^.*[\d].*$")
int_prefixes = "s|st|th|nd|rd"
all_prefixes = "km|m|v|K|b|bn|billion|k|million|th\+"
careful_num_regex = re.compile("^\+?"
                               "(\d{1,3}(,\d{3})*|\d+|(?=\.))"
                               "(?:(\.\d+)?(?P<p1>%s)?|(?P<p2>%s)?)\+?$" % (all_prefixes, int_prefixes))


def is_number(token):
    match = careful_num_regex.fullmatch(token)
    if match is None:
        return None
    p1 = match.group("p1")
    p2 = match.group("p2")
    if p1 is not None:
        return p1
    elif p2 is not None:
        return p2
    else:
        return ""


class QaTextFeautrizer(Configurable):

    def n_context_features(self):
        raise NotImplementedError()

    def n_question_features(self):
        raise NotImplementedError()

    def get_features(self, question, context):
        """
        return arrays of shape (n_question_words, feature_dim) (n_context_words, feature_dim)
        """
        raise NotImplementedError()


class BasicWordFeatures(QaTextFeautrizer):
    features_names = ["Num", "NumPrefix", "NumExp", "AnyNum", "Punct",
                      "Cap", "Upper", "Alpha", "NonEng", "Len"]

    def __init__(self):
        self.any_num_regex = re.compile("^.*\d.*$")
        self.num_exp = re.compile("^[\d+x\-/\\\=\u2013,:\W]*$")
        self.punc_regex = re.compile("^\W+$")
        self.alpha = re.compile("^[a-z]+$")
        self.any_non_english = re.compile(".*[^a-zA-Z0-9\W].*")
        self.non_english = re.compile("^[^a-zA-Z0-9\W]+$")
        self._feature_cache = {}

    def get_word_features(self, word):
        if word not in self._feature_cache:
            num_prefix = is_number(word)
            non_eng = self.non_english.match(word) is not None
            punc = self.punc_regex.match(word) is not None
            features = np.array([
                num_prefix is not None,
                num_prefix is not None and num_prefix != "",
                self.num_exp.match(word) is not None and num_prefix is None and not punc,
                self.any_num_regex.match(word) is not None and not punc,
                punc,
                word[0].isupper() and word[1:].islower() and not non_eng,
                word.isupper() and not non_eng,
                self.alpha.match(word) is not None,
                non_eng,
                np.log(len(word))
            ])
            self._feature_cache[word] = features
            return features
        return self._feature_cache[word]

    @property
    def n_features(self):
        return 10

    def n_context_features(self):
        return self.n_features

    def n_question_features(self):
        return self.n_features

    def get_sentence_features(self, sent):
        features = np.zeros((len(sent), self.n_features))
        for i, word in enumerate(sent):
            features[i, :self.n_features] = self.get_word_features(word)
        return features

    def get_features(self, question, context):
        return self.get_sentence_features(question), self.get_sentence_features(context)


def extract_year(token):
    ends_with_s = False
    if token[-1] == "s":
        token = token[:-1]
        ends_with_s = True
    try:
        val = int(token)
        if val < 100 and val % 10 == 0 and ends_with_s:
            return 1900 + val
        if 1000 <= val <= 2017:
            return val
        return None
    except ValueError:
        return None


class MatchWordFeatures(QaTextFeautrizer):
    def __init__(self, require_unique_match, lemmatizer="word_net",
                 empty_question_features=False, stop_words=None):
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
        self.empty_question_features = empty_question_features
        if lemmatizer == "word_net":
            self._lemmatizer = WordNetLemmatizer()
        else:
            raise ValueError()
        self._cache = {}
        self.require_unique_match = require_unique_match

    def n_context_features(self):
        return 3

    def n_question_features(self):
        return 3 if self.empty_question_features else 0

    def lemmatize_word(self, word):
        cur = self._cache.get(word)
        if cur is None:
            cur = self._lemmatizer.lemmatize(word)
            self._cache[word] = cur
        return cur

    def get_features(self, question, context):
        stop = set() if self.stop_words is None else self.stop_words.words
        context_features = np.zeros((len(context), 3))

        if not self.require_unique_match:
            question_words = set(x for x in question if x.lower() not in stop)
            quesiton_words_lower = set(x.lower() for x in question)
            quesiton_words_stem = set(self.lemmatize_word(x) for x in quesiton_words_lower)
        else:
            question_words = set(k for k,v in Counter(question).items() if v == 1)
            quesiton_words_lower = set(k for k,v in Counter(x.lower() for x in question_words).items() if v == 1)
            quesiton_words_stem = set(k for k, v in Counter(self.lemmatize_word(x) for x
                                                            in quesiton_words_lower).items() if v == 1)

        for i, word in enumerate(context):
            if word in question_words:
                context_features[i][:3] = 1
            elif word.lower() in quesiton_words_lower:
                context_features[i][:2] = 1
            elif self._lemmatizer.lemmatize(word) in quesiton_words_stem:
                context_features[i][2] = 1

        if self.empty_question_features:
            return np.zeros((len(question), 3)), context_features
        else:
            return np.zeros((len(question), 0)), context_features

    def __setstate__(self, state):
        self.__init__(**state)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_cache"]
        del state["_lemmatizer"]
        return state

