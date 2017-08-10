import re
import string
from typing import List, Tuple
from collections import Counter

import unicodedata
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from configurable import Configurable

extra_split_chars = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0")
extra_split_tokens = ("``", "(?<=[^_])_(?=[^_])", "''", "[" + "".join(extra_split_chars) + "]")
extra_split_chars_re = re.compile("(" + "|".join(extra_split_tokens) + ")")
double_quote_re = re.compile("\"|``|''")
space_re = re.compile("[ \u202f]")


def post_split_tokens(tokens: List[str]) -> List[str]:
    """ Apply a small amount of extra splitting to the given tokens, this is in particular to avoid UNK tokens
     due to contraction, quotation, or other forms of puncutation. """
    resplit_sent = []
    for token in tokens:
        resplit_sent += [x for x in extra_split_chars_re.split(token) if x != ""]
    return resplit_sent


def our_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r"\s*[\u2212°%£€¥\u2000-\u206F\u2E00-\u2E7F\\'!\"#$%&()*+,\u2013\-.\/:;<=>?@\[\]^_`{|}~]\s*", '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(clean_text(s)))))


def our_text_score(prediction, ground_truth):
    prediction_tokens = our_normalize_answer(prediction).split()
    ground_truth_tokens = our_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, prediction_tokens == ground_truth_tokens


def clean_text(text):
    # be consistent with quotes, and replace \u2014 and \u2212 which I have seen being mapped to UNK
    # by glove word vecs with different characters
    return text.replace("''", "\"").replace("``", "\"").replace("\u2212", "-").replace("\u2014", "\u2013")


def convert_to_spans(raw_text: str, sentences: List[List[str]]) -> List[List[Tuple[int, int]]]:
    """ Convert a tokenized version of `raw_text` into a series character spans referencing the `raw_text` """
    cur_idx = 0
    all_spans = []
    for sent in sentences:
        spans = []
        for token in sent:
            # Tokenizer might transform double quotes, for these case search over several
            # possible encodings
            if double_quote_re.match(token):
                span = double_quote_re.search(raw_text[cur_idx:])
                tmp = cur_idx + span.start()
                l = span.end() - span.start()
            else:
                tmp = raw_text.find(token, cur_idx)
                l = len(token)
            if tmp < cur_idx:
                raise ValueError(token)
            cur_idx = tmp
            spans.append((cur_idx, cur_idx + l))
            cur_idx += l
        all_spans.append(spans)
    return all_spans


def get_word_span(spanss: List[List[Tuple[int, int]]], start: int, stop: int) -> List[Tuple[int, int]]:
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if span[1] > start:
                if span[0] < stop:
                    idxs.append((sent_idx, word_idx))
                else:
                    break
    return idxs


def get_paragraph_tokenizer(tokenizer_name):
    if tokenizer_name == "NLTK_AND_CLEAN":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens: str) -> List[str]:
            tokens = nltk.word_tokenize(tokens)
            return [clean_text(token) for token in post_split_tokens(tokens)]
        return sent_tokenize, word_tokenize
    elif tokenizer_name == "NLTK":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens: str) -> List[str]:
            tokens = nltk.word_tokenize(tokens)
            return [token for token in post_split_tokens(tokens)]
        return sent_tokenize, word_tokenize
    else:
        raise ValueError()


class WordNormalizer(Configurable):
    def __init__(self, lower: bool = True, stemmer="port"):
        self.lower = lower
        self.stemmer = stemmer
        if stemmer == "port":
            self._stemmer = PorterStemmer()
            self._stem = self._stemmer.stem
        elif stemmer == "wordnet":
            self._stemmer = WordNetLemmatizer()
            self._stem = self._stemmer.lemmatize
        else:
            raise ValueError(stemmer)
        # stemming is slow, so we cache words as we go
        self.normalize_cache = {}

    def normalize(self, x):
        if self.lower:
            x = x.lower()

        norm = self.normalize_cache.get(x)

        if norm is not None:
            return norm

        # Sometimes questions have "ascii versions" of fancy unicode text in the context, so
        # attempt ascii-convert characters when normalized
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode("ascii")
        if self.lower or x.islower():
            stem = self._stem(x)
        else:
            stem = x

        self.normalize_cache[x] = stem
        return stem

    def __getstate__(self):
        return dict(lower=self.lower, stemmer=self.stemmer)

    def __setstate__(self, state):
        self.__init__(**state)


class NltkPlusStopWords(Configurable):
    """ Configurablable access to stop word """

    def __init__(self, punctuation=False):
        self._words = None
        self.punctuation = punctuation

    @property
    def words(self):
        if self._words is None:
            self._words = set(stopwords.words('english'))
            self._words.update(["many", "how", "de"])
            if self.punctuation:
                self._words.update(string.punctuation)
                self._words.update(["£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0"])
        return self._words

    def __getstate__(self):
        return dict(punctuation=self.punctuation)

    def __setstate__(self, state):
        self.__init__(**state)
