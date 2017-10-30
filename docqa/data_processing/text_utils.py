import re
import string
import unicodedata
from collections import Counter
from typing import List, Tuple

import nltk
import numpy as np
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from docqa.utils import flatten_iterable

from docqa.configurable import Configurable

extra_split_chars = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013", "/", "~",
                     '"', "'", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0")
extra_split_tokens = ("``",
                      "(?<=[^_])_(?=[^_])",  # dashes w/o a preceeding or following dash, so __wow___ -> ___ wow ___
                      "''", "[" + "".join(extra_split_chars) + "]")
extra_split_chars_re = re.compile("(" + "|".join(extra_split_tokens) + ")")
double_quote_re = re.compile("\"|``|''")
space_re = re.compile("[ \u202f]")


def post_split_tokens(tokens: List[str]) -> List[str]:
    """
    Apply a small amount of extra splitting to the given tokens, this is in particular to avoid UNK tokens
    due to contraction, quotation, or other forms of puncutation. I haven't really done tests to see
    if/how much difference this makes, but it does avoid some common UNKs I noticed in SQuAD/TriviaQA
     """
    return flatten_iterable([x for x in extra_split_chars_re.split(token) if x != ""]
                            for token in tokens)


def get_word_span(spans: np.ndarray, start: int, stop: int):
    idxs = []
    for word_ix, (s, e) in enumerate(spans):
        if e > start:
            if s < stop:
                idxs.append(word_ix)
            else:
                break
    return idxs


class ParagraphWithInverse(object):
    """
    Paragraph that retains the inverse mapping of tokens -> span in the original text,
    Used if we want to get the untokenized, uncleaned text for a particular span
    """

    @staticmethod
    def empty():
        return ParagraphWithInverse([], "", np.zeros((0, 2), dtype=np.int32))

    @staticmethod
    def concat(paras: List, delim: str):
        paras = [x for x in paras if x.n_tokens > 0]
        original_text = delim.join([x.original_text for x in paras])
        full_inv = []
        all_tokens = []
        on_char = 0
        for para in paras:
            if para.n_tokens == 0:
                continue
            all_tokens += para.text
            full_inv.append(para.spans + on_char)

            on_char += para.spans[-1][1] + len(delim)
        if len(all_tokens) == 0:
            return ParagraphWithInverse.empty()
        return ParagraphWithInverse(all_tokens, original_text, np.concatenate(full_inv))

    def __init__(self, text: List[List[str]], original_text: str, spans: np.ndarray):
        if spans.shape != (sum(len(s) for s in text), 2):
            raise ValueError("Spans should be shape %s but got %s" % ((sum(len(s) for s in text), 2), spans.shape))
        self.text = text
        self.original_text = original_text
        self.spans = spans

    def get_context(self):
        return flatten_iterable(self.text)

    def get_original_text(self, start, end):
        """ Get text between the token at `start` and `end` inclusive """
        return self.original_text[self.spans[start][0]:self.spans[end][1]]

    @property
    def n_tokens(self):
        return sum(len(s) for s in self.text)


class NltkAndPunctTokenizer(Configurable):

    @staticmethod
    def convert_to_spans(raw_text: str, sentences: List[List[str]]) -> List[List[Tuple[int, int]]]:
        """ Convert a tokenized version of `raw_text` into a series character spans referencing the `raw_text` """
        cur_idx = 0
        all_spans = []
        for sent in sentences:
            spans = []
            for token in sent:
                # (our) Tokenizer might transform double quotes, for this case search over several
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

    def __init__(self):
        self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
        self.word_tokenizer = nltk.TreebankWordTokenizer()

    def clean_text(self, word):
        # be consistent with quotes, and replace \u2014 and \u2212 which I have seen being mapped to UNK
        # by glove word vecs
        return word.replace("''", "\"").replace("``", "\"").replace("\u2212", "-").replace("\u2014", "\u2013")

    def tokenize_sentence(self, sent) -> List[str]:
        tokens = self.word_tokenizer.tokenize(sent)
        return [self.clean_text(token) for token in post_split_tokens(tokens)]

    def tokenize_paragraph(self, paragraph: str) -> List[List[str]]:
        return [self.tokenize_sentence(s) for s in self.sent_tokenzier.tokenize(paragraph)]

    def tokenize_paragraph_flat(self, paragraph: str) -> List[str]:
        return flatten_iterable(self.tokenize_paragraph(paragraph))

    def tokenize_with_inverse(self, paragraph: str, is_sentence: bool=False) -> ParagraphWithInverse:
        if is_sentence:
            para = [paragraph]
        else:
            para = self.sent_tokenzier.tokenize(paragraph)

        text = [self.word_tokenizer.tokenize(s) for s in para]
        for i, sent in enumerate(text):
            text[i] = post_split_tokens(sent)

        # recover (start, end) for each token relative to the raw `context`
        text_spans = self.convert_to_spans(paragraph, text)

        # Clean text, we do this at the end so `convert_to_spans` works as expected
        for i, sent in enumerate(text):
            text[i] = [self.clean_text(x) for x in sent]

        text_spans = flatten_iterable(text_spans)
        if len(text_spans) == 0:
            text_spans = np.zeros((0, 2), dtype=np.int32)
        else:
            text_spans = np.array(text_spans, dtype=np.int32)
        return ParagraphWithInverse(text, paragraph, text_spans)


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
            # Common question words we probably want to ignore, "de" was suprisingly common
            # due to its appearance in person names
            self._words.update(["many", "how", "de"])
            if self.punctuation:
                self._words.update(string.punctuation)
                self._words.update(["£", "€", "¥", "¢", "₹", "\u2212",
                                    "\u2014", "\u2013", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0"])
        return self._words

    def __getstate__(self):
        return dict(punctuation=self.punctuation)

    def __setstate__(self, state):
        self.__init__(**state)


WEEK_DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December",
          "Jan.", "Feb.", "Mar.", "Jul.", "Jun.", "Apr.", "Aug.", "Sept.", "Sep.", "Oct.", "Nov.", "Dec.",
          "Jan",  "Feb",  "Mar",  "Jul",  "Jun",  "Apr",  "Aug",  "Sept",  "Sep",  "Oct",  "Nov",  "Dec"}
HONORIFIC = {"Ltd", "Lt", "Sgt", "Sr", "Jr", "Mr", "Mrs", "Ms", "Dr",
             "Ltd.", "Lt.", "Sgt.", "Sr.", "Jr.", "Mr.", "Mrs.", "Ms.", "Dr.",
             "Miss", "Madam", "Sir", "Majesty", "Saint", "Prof", "Private"}
TITLE = {"Princess", "King", "Queen", "Prince", "Duke", "Lord", "Lady",
         "Archduke", "Archduchess", "Earl", "Baron", "Baroness", "Marquis",  "Marquess",
         "Senator", "Representative", "Count", "Countess", "Viscount", "Viscountess",
         "Emperor", "Empress", "Viceroy", "Pope", "Pastor", "Cardinal", "Priest",
         "President", "Mayor", "Governor", "Admiral", "Captain", "Colonel", "Major",
         "Sergeant", "Judge", "Mama", "Papa"}
PLANETS = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"}


class NameDetector(Configurable):
    squad_exceptions = {
        "Congress", "Senate", "Olympic", "St", "Bible", "Android", "Bactria",
        "Nobel", "Mount", "Excalibur", "Internationale",
        "Coke", "Pepsi", "Google", "Mac",
        "Facebook", "Twitter", "Wikipedia", "Google", "Amazon", "Airways",
        "Irreplaceable", "Privy", "Def", "Prev", "Bactrian", "Fe", "Boo",
        "Bang", "Atlas", "Corp", "Academy",
        "Co.", "Co", "Inc.", "Inc", "Hz", "St."}

    @property
    def version(self):
        # Expanded words
        return 1

    def __init__(self):
        self.stop = None
        self.word_counts = None
        self.word_counts_lower = None

    def init(self, word_counts):
        print("Loading...")
        stop = set(self.squad_exceptions)
        stop.update(stopwords.words('english'))
        stop.update(TITLE)
        stop.update(HONORIFIC)
        stop.update(PLANETS)
        stop.update(WEEK_DAYS)
        stop.update(x + "s" for x in WEEK_DAYS)
        stop.update(MONTHS)
        self.stop = {x.lower() for x in stop}
        self.word_counts = word_counts
        word_counts_lower = Counter()
        for k, v in word_counts.items():
            word_counts_lower[k.lower()] += v
        self.word_counts_lower = word_counts_lower

    def select(self, word):
        if word[0].isupper() and word[1:].islower() and len(word) > 1:
            wl = word.lower()
            if wl not in self.stop:
                lc = self.word_counts_lower[wl]
                if lc == 0 or (self.word_counts[word] / lc) > 0.9:
                    return True
        return False

