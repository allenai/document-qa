import string
from typing import List
import numpy as np
from os.path import exists
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from data_processing.text_features import is_number
from data_processing.text_utils import get_paragraph_tokenizer
from trivia_qa.build_span_corpus import TriviaQaSampleWebDataset
from utils import flatten_iterable


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


NORMALIZED_CACHE = "/tmp/normalized_titles.txt"


def get_normalized_titles():
    if exists(NORMALIZED_CACHE):
        with open(NORMALIZED_CACHE, "r") as f:
            return set(tuple(x.strip().split(" ")) for x in f)

    with open("/Users/chrisc/Desktop/tmp/enwiki-latest-all-titles-in-ns0", "r") as f:
        titles = list(x.strip() for x in f)
    titles = [x.lower().replace("_", " ") for x in titles]
    word_tokenizer = get_paragraph_tokenizer("NLTK")[1]
    strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])
    normalized_titles = set()
    for title in tqdm(titles):
        tokens = tuple(x.strip(strip) for x in word_tokenizer(title))
        normalized_titles.add(tuple(x for x in tokens if x != ""))
    with open("/tmp/normalized_titles.txt", "w") as f:
        for title in normalized_titles:
            f.write(" ".join(title))
            f.write("\n")
    return normalized_titles


class AnswerCandidateTagger(object):
    def tag_entities(self, text: List[str]) -> np.ndarray:
        raise NotImplementedError()


class ExactMatchTagger(AnswerCandidateTagger):
    def __init__(self):
        with open("/Users/chrisc/Desktop/tmp/enwiki-latest-all-titles-in-ns0", "r") as f:
            titles = list(x.strip() for x in f)
        titles = {x.lower().replace("_", " ") for x in titles}
        self.titles = titles

    def tag_entities(self, text: List[str]) -> np.ndarray:
        anwers_candidates = []
        for s in range(len(text)):
            end = min(s + 8, len(text))
            for e in range(s + 1, end):
                if " ".join(text[s:e]).lower() in self.titles:
                    anwers_candidates.append((s, e-1))
        return np.array(anwers_candidates)


class NormalizedMatchTagger(AnswerCandidateTagger):
    def __init__(self):
        self.titles = get_normalized_titles()
        self.prefixes = set()
        for title in self.titles:
            for l in range(1, len(title)):
                self.prefixes.add(title[0:l])
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])
        self._units = get_units()


    def tag_entities(self, text: List[str]) -> np.ndarray:
        cleaned = [x.lower().strip(self.strip) for x in text]
        anwers_candidates = []
        for s in range(len(text)):
            token = cleaned[s]
            if token == "":
                continue
            tokens = (token, )
            is_num = is_number(token)
            if is_num:
                anwers_candidates.append((s, s))
                for i in range(s+1, min(s+4, len(text))):
                    if " ".join(cleaned[s+1:i]) in self._units:
                        anwers_candidates.append((s, s+i+1))
                if len(text) < s - 1:
                    next = tokens[s + 1]
                    # if next in {"times", "feet", "yards", "inches", ""}
            if not (is_num or tokens in self.prefixes or token[0].isupper()):
                continue
            if len(text) == s - 1:
                if tokens in self.titles or is_num:
                    anwers_candidates.append((s, s))
                continue

            tokens = [tokens[0]]
            next = text[s + n]
            # while n < len(text)

                # if is_number(token):
                #     tokens.append(token)

            end = min(s + 8, len(text))
            for e in range(s + 1, end):
                if tuple(x for x in cleaned[s:e] if x != "") in self.titles:
                    anwers_candidates.append((s, e-1))
        return np.array(anwers_candidates)


def get_units():
    unit = wn.synsets("unit_of_measurement")[0]

    def get_hyponyms(synset):
        words = set()
        for hyponym in synset.hyponyms():
            words |= set(get_hyponyms(hyponym))
        return words | set(x.name().lower().replace("_", " ") for x in synset.lemmas())

    return get_hyponyms(unit)

def show_entities():
    # <in|in the> <entity>
    # num <miles|times|feet>
    # <x> and <y>
    # Allow: (in|the|-|Caps|

    # print("Loading titles")
    # print(len(get_normalized_titles()))
    # return
    tagger = NormalizedMatchTagger()
    data = TriviaQaSampleWebDataset()
    n_candidates = 0
    found = 0
    print("Loading train")
    points = data.get_train()
    points = flatten_iterable([(q, doc) for doc in q.all_docs] for q in points)
    np.random.RandomState(0).shuffle(points)
    points = points[:10000]
    for question, doc in points:
        answers = doc.answer_spans[doc.answer_spans[:, 1] < 400]
        if len(answers) == 0:
            continue
        answers = set((s,e) for s, e in answers)
        text = data.evidence.get_document(doc.doc_id, n_tokens=400, flat=True)
        candidates = tagger.tag_entities(text)
        if any((s,e) in answers for s,e in candidates):
            found += 1
        else:
            print(" ".join(question.question))
            print(list(set(" ".join(text[s:e+1]) for s,e in doc.answer_spans)))
            tagger.tag_entities(text)
        n_candidates += 1

    print("Found %d/%d (%.4f)" % (found, n_candidates, found/n_candidates))

if __name__ == "__main__":
    print(get_units())
    # show_entities()