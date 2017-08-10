from collections import defaultdict
from typing import List

import nltk
import numpy as np
from nltk.corpus import stopwords

from configurable import Configurable
from squad.build_squad_data import SquadCorpus
from utils import flatten_iterable


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


class Extractor(Configurable):

    def __init__(self):
        self.stop = set(x for x in stopwords.words('english'))
        add = ["'", ",", "-", "\"", "?", "many", "happens", "happened", "usually",
               "defined", "year", "examples", "date", "first", "formed", "another",
               "according", "much", ")", "(", "th", "basis"]
        self.tagger = nltk.PerceptronTagger()
        self.stop.update(add)

    def extract(self, question: List[str]):
        tagged = self.tagger.tag(question)
        key_words = []
        for word, pos_tag in tagged:
            if word.lower() not in self.stop:
                # if pos_tag in ["NNP", "NN", "NNS", "CD", "JJ"] or word[0].isupper():
                key_words.append(word)
        return key_words


def show_matches(questions_per_doc, extractor):
    docs = SquadCorpus().get_train_docs()
    for doc in docs:
        word_to_para = defaultdict(list)
        word_to_sent = defaultdict(list)

        for para in doc.paragraphs:
            para_words = set()
            for sent in para.context:
                sent_words = set(x.lower() for x in sent)
                for word in sent_words:
                    word_to_sent[word].append((para.paragraph_num, sent))
                para_words.update(sent_words)
            for word in para_words:
                word_to_para[word].append(para.paragraph_num)

        questions = flatten_iterable([[(q, x.paragraph_num) for q in x.questions] for x in doc.paragraphs])
        np.random.shuffle(questions)
        questions = questions[:questions_per_doc]
        for question, para_num in questions:
            print(" ".join(question.words))
            key_words = extractor.extract(question.words)
            print(["%s (%d)" % (k, len(word_to_para[k.lower()])) for k in key_words])
            para_matches = set(flatten_iterable([word_to_para[x.lower()] for x in key_words]))
            print("%d / %d paragraphs match" % (len(para_matches), len(doc.paragraphs)))


if __name__ == "__main__":
    show_matches(10, Extractor())