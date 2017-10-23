import re
from collections import Counter, defaultdict

import numpy as np

from docqa.data_processing.text_features import BasicWordFeatures
from docqa.squad.squad_data import SquadCorpus


def show_unk(corpus: SquadCorpus, vec_name: str,
             context: bool=True, question: bool=True):
    vecs = corpus.get_pruned_word_vecs(vec_name)
    docs = corpus.get_train()

    lower_unk = Counter()
    unk = Counter()

    for doc in docs:
        for para in doc.paragraphs:
            if context:
                for sent in para.text:
                    for word in sent:
                        if word not in vecs:
                            unk[word] += 1
                        word = word.lower()
                        if word not in vecs:
                            lower_unk[word] += 1
            if question:
                for question in para.questions:
                    for word in question.words:
                        if word not in vecs:
                            unk[word] += 1
                        word = word.lower()
                        if word not in vecs:
                            lower_unk[word] += 1

    print("\n".join("%s: %d" % (k,v) for k,v in lower_unk.most_common()))


def show_features(corpus: SquadCorpus, vec_name):
    print("Loading train docs")
    data = corpus.get_train()
    np.random.shuffle(data)
    data = data[:100]

    print("Loading vectors")
    vecs = corpus.get_pruned_word_vecs(vec_name)
    fe = BasicWordFeatures()

    grouped_by_features = defaultdict(Counter)

    print("start")

    for doc in data:
        paragraphs = list(doc.paragraphs)
        np.random.shuffle(paragraphs)
        for para in paragraphs:
            sentences = list(para.text) + [x.words for x in para.questions]
            np.random.shuffle(sentences)
            for words in sentences:
                for i, word in enumerate(words):
                    if word.lower() not in vecs:
                        x = fe.get_word_features(word)
                        for i, val in enumerate(x):
                            if val > 0:
                                grouped_by_features[i][word] += 1

    for i in sorted(grouped_by_features.keys()):
        name = BasicWordFeatures.features_names[i]
        if name in ["Len"]:
            continue
        vals = grouped_by_features[i]
        print()
        print("*"*30)
        print("%s-%d %d (%d)" % (name, i, len(vals), sum(vals.values())))
        for k,v in vals.most_common(30):
            print("%s: %d" % (k, v))


def show_nums(corpus: SquadCorpus):
    n_regex = re.compile(".*[0-9].*")
    data = corpus.get_train()
    np.random.shuffle(data)

    for doc in data:
        paragraphs = list(doc.paragraphs)
        np.random.shuffle(paragraphs)
        for para in paragraphs:
            sentences = list(para.context) + [x.words for x in para.questions]
            np.random.shuffle(sentences)
            for words in sentences:
                for i, word in enumerate(words):
                    if n_regex.match(word) is not None:
                        print(word)


def show_in_context_unks(corpus: SquadCorpus, vec_name):
    data = corpus.get_train()
    np.random.shuffle(data)
    vecs = corpus.get_pruned_word_vecs(vec_name)

    for doc in data:
        paragraphs = list(doc.paragraphs)
        np.random.shuffle(paragraphs)
        for para in paragraphs:
            sentences = list(para.text) + [x.words for x in para.questions]
            np.random.shuffle(sentences)
            for words in sentences:
                for i, word in enumerate(words):
                    if word.lower() not in vecs:
                        words[i] = "{{{" + word + "}}}"
                        print(" ".join(words[max(0,i-10):min(len(words),i+10)]))
                        words[i] = word


def main():
    show_unk(SquadCorpus(), "glove.840B.300d")
    # show_unk(SpanCorpus("squad"), "glove.6B.100d")
    # show_nums(SpanCorpus("squad"))


if __name__ == "__main__":
    main()