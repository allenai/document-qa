from collections import Counter

import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors

from data_processing.document_splitter import TopTfIdf, MergeParagraphs
from data_processing.paragraph_qa import split_docs
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import QaCorpusLazyStats, compute_voc, ParagraphAndQuestionSpec
from data_processing.text_features import is_number
from data_processing.text_utils import WEEK_DAYS, MONTHS, NameDetector, NltkPlusStopWords
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from nn.embedder import DropNames, DropNamesV2
from squad.build_squad_dataset import SquadCorpus
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.triviaqa_training_data import ExtractSingleParagraph, InMemoryWebQuestionBuilder
from utils import flatten_iterable


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
    print("Loading...")
    squad = False
    if squad:
        corpus = SquadCorpus()
        docs = corpus.get_train()
        data = split_docs(docs)
    else:
        stop = NltkPlusStopWords()
        data = PreprocessedData(TriviaQaWebDataset(),
                                ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), intern=True),
                                InMemoryWebQuestionBuilder(None, None),
                                eval_on_verified=False
                                )
        data.load_preprocess("triviaqa-web-merge400-tfidf1.pkl.gz")
        data = data.get_train().data
    print("Get voc...")

    detector = NameDetector()
    wc = QaCorpusLazyStats(data).get_word_counts()
    detector.init(wc)

    is_named = 0
    context_names = 0
    np.random.shuffle(data)
    for point in data:
        context = flatten_iterable(point.context)
        for s,e in point.answer.answer_spans[0:1]:
            for i in range(s, e+1):
                if detector.select(context[i]):
                    is_named += 1
                    break
            # print(" ".join(point.question))
            tmp = list(context)
            c_s, c_e = max(s - 10, 0), min(e + 10, len(context))
            for i in range(c_s, c_e):
                if detector.select(tmp[i]):
                    context_names += 1
                    # if tmp[i] in vecs:
                    # tmp[i] = "<" + tmp[i] + ">"
                    # else:
                    #     tmp[i] = "<?" + tmp[i] + ">"
            #
            # tmp[s] = "{{{" + tmp[s]
            # tmp[e] = tmp[e] + "}}}"
            # print(" ".join(tmp[c_s:c_e]))
            # input()
            # for word in context[s:e+1]:
            #     if detector.is_name(word):
            #         answer_counts[word] += 1
    # for k, v in answer_counts.most_common(500):
    #     print("%s: %d" % (k, v))
    # print("%d / %d (%.4f)" % (c, len(data), c/len(data)))
    print("%d / %d (%.4f)" % (is_named, len(data), is_named/len(data)))
    print("%d / %d (%.4f)" % (context_names, len(data), context_names / len(data)))


def show_names():
    # corpus = SquadCorpus()
    print("Loading...")
    stop = NltkPlusStopWords()
    data = PreprocessedData(TriviaQaWebDataset(),
                            ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), intern=True),
                            InMemoryWebQuestionBuilder(None, None),
                            eval_on_verified=False
                            )
    data.load_preprocess("triviaqa-web-merge400-tfidf1.pkl.gz")
    data = data.get_train().data

    print("Word counts")
    wc = QaCorpusLazyStats(data).get_word_counts()

    print("Init...")
    Counter().most_common()
    # data = split_docs(corpus.get_train())
    detector = NameDetector()
    detector.init(wc)

    ix = 0
    for k, c in wc.most_common():
        if detector.select(k):
            print("%s: %d" % (k ,c))
            ix += 1
            if ix > 5000:
                break


def main():
    embed = DropNamesV2(vec_name="glove.840B.300d",
                        selector=NameDetector(),
                        word_vec_init_scale=0, learn_unk=False,
                        keep_probs=0, kind="shuffle")
    corpus = SquadCorpus()
    squad = False
    print("Loading...")
    if squad:
        docs = corpus.get_train()
        data = split_docs(docs)
    else:
        stop = NltkPlusStopWords()
        data = PreprocessedData(TriviaQaWebDataset(),
                                ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), intern=True),
                                InMemoryWebQuestionBuilder(None, None),
                                eval_on_verified=False
                                )
        data.load_preprocess("triviaqa-web-merge400-tfidf1.pkl.gz")
        data = data.get_train().data
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
    encoder.init(ParagraphAndQuestionSpec(1, None, None, None), False, embed, None)

    # sess = tf.Session()

    np.random.shuffle(data)
    for q in data:
        encoded = encoder.encode([q], True)
        print(q.question)
        context_words, question_words = [encoded[encoder.context_words], encoded[encoder.question_words]]
        print([ix_to_word[i] for i in question_words[0]])
        # context_words, question_words = embed.drop_names([context_words, question_words])
        # print([ix_to_word[i] for i in sess.run(question_words)[0]])


if __name__ == "__main__":
    # show_names()
    show_answers()