from collections import Counter

import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors

from data_processing.paragraph_qa import split_docs
from data_processing.qa_data import QaCorpusLazyStats, compute_voc, ParagraphAndQuestionSpec
from data_processing.text_features import is_number
from data_processing.text_utils import WEEK_DAYS, MONTHS, NameDetector
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from nn.embedder import DropNames
from squad.build_squad_dataset import SquadCorpus
from utils import flatten_iterable


def is_named(self, word):
    if word[0].isupper() and word[1:].islower() and len(word) > 1:
        wl = word.lower()
        if wl not in self._stop:
            lc = self._word_counts_lower[wl]
            if lc == 0 or (self._word_counts[word] / lc) > self.named_thresh:
                return True
    return False


def show_nn():
    corpus = SquadCorpus()
    print("Load train")
    data = split_docs(corpus.get_train())
    print("Comput stats")
    wc = QaCorpusLazyStats(data).get_word_counts()
    detector = NameDetector(wc)
    print("Load vecs")
    vecs = corpus.get_resource_loader().load_word_vec("glove.840B.300d")

    print('Scanning...')
    names = Counter()
    for word, c in wc.items():
        if detector.is_name(word):
            names[word] = c
    vec_names = [k for k in names if k in vecs]
    print("Have vec for %d/%d (%.4f)" % (len(vec_names), len(names), len(vec_names)/len(names)))

    print("Fit nn")
    nn = NearestNeighbors()
    mat = np.stack([vecs[k] for k in vec_names], axis=0)
    print(mat.shape)
    nn.fit(mat)

    ixs = np.arange(0, len(vec_names))
    np.random.shuffle(ixs)
    for ix in ixs:
        print(vec_names[ix])
        dist, ix = nn.kneighbors(mat[ix:ix+1])
        print([("%s" % vec_names[i]) for i in ix[0]])




def show_answers():
    corpus = SquadCorpus()
    print("Loading...")
    data = split_docs(corpus.get_train())
    detector = NameDetector()
    detector.init(QaCorpusLazyStats(data).get_word_counts())
    answer_counts = Counter()
    c = 0
    np.random.shuffle(data)
    for point in data:
        context = flatten_iterable(point.context)
        for s,e in point.answer.answer_spans:
            print(" ".join(point.question))
            tmp = list(context)
            c_s, c_e = max(s - 10, 0), min(e + 10, len(context))
            for i in range(c_s, c_e):
                if detector.select(tmp[i]):
                    tmp[i] = "[[" + tmp[i] + "]]"
            tmp[s] = "{{{" + tmp[s]
            tmp[e] = tmp[e] + "}}}"
            print(" ".join(tmp[c_s:c_e]))
            input()
            # for word in context[s:e+1]:
            #     if detector.is_name(word):
            #         answer_counts[word] += 1
    # for k, v in answer_counts.most_common(500):
    #     print("%s: %d" % (k, v))
    # print("%d / %d (%.4f)" % (c, len(data), c/len(data)))



def show_names():
    corpus = SquadCorpus()
    print("Loading...")
    data = split_docs(corpus.get_train())
    detector = NameDetector()
    detector.init(QaCorpusLazyStats(data).get_word_counts())


def main():
    embed = DropNames(vec_name="glove.840B.300d",
                      selector=NameDetector(),
                      word_vec_init_scale=0, learn_unk=False,
                      keep_probs=0, kind="shuffle")
    corpus = SquadCorpus()
    print("Loading...")
    docs = corpus.get_train()
    data = split_docs(docs)
    print("Get voc...")
    voc = compute_voc(data)
    print("Init ...")
    stats = QaCorpusLazyStats(data)
    loader = corpus.get_resource_loader()
    embed.set_vocab(stats, loader, [])
    embed.init(loader, voc)

    print("Init encoder")

    ix_to_word = {ix:w for w, ix in embed._word_to_ix.items()}
    ix_to_word[1] = "UNK"
    ix_to_word[0] = "PAD"

    encoder = DocumentAndQuestionEncoder(SingleSpanAnswerEncoder())
    encoder.init(ParagraphAndQuestionSpec(1, None, None, None, None, None), False, embed, None)

    sess = tf.Session()

    np.random.shuffle(data)
    for q in data:
        encoded = encoder.encode([q], True)
        context_words, question_words = [encoded[encoder.context_words], encoded[encoder.question_words]]
        print([ix_to_word[i] for i in question_words[0]])
        context_words, question_words = embed.drop_names([context_words, question_words])
        print([ix_to_word[i] for i in sess.run(question_words)[0]])


if __name__ == "__main__":
    show_names()