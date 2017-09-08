import json
from collections import Counter
import re
import numpy as np
from os.path import join

from tqdm import tqdm

import config
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.multi_paragraph_qa import RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey, ParagraphAndQuestionDatasetBuilder
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder
from evaluator import LossEvaluator
from experimental.WithTfIdfFeatures import ExtractSingleParagraphFeaturized
from nn.embedder import FixedWordEmbedder, LearnedCharEmbedder
from squad.squad_data import SquadCorpus
from text_preprocessor import WithIndicators
from trivia_qa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWebDataset
from trivia_qa.training_data import ExtractMultiParagraphs, ExtractSingleParagraph
from trivia_qa.triviaqa_evaluators import BoundedSpanEvaluator

import tensorflow as tf

from utils import flatten_iterable, ResourceLoader


def main1():
    data = SquadCorpus()
    data.dir = join(config.CORPUS_DIR, "squad-v3")
    data2 = SquadCorpus()
    data2.dir = join(config.CORPUS_DIR, "squad-v2")

    train = data.get_train()
    train2 = data2.get_train()
    if len(train) != len(train2):
        raise ValueError()

    for d1, d2 in zip(train, train2):
        if d1.doc_id != d2.doc_id or d1.title != d2.title:
            raise ValueError()
        if len(d1.paragraphs) != len(d2.paragraphs):
            raise ValueError()
        for p1, p2 in zip(d1.paragraphs, d2.paragraphs):
            if p1.context != p2.context or p1.paragraph_num != p2.paragraph_num or p1.original_text != p2.original_text:
                raise ValueError()
            if not np.all(p1.spans == p2.spans):
                raise ValueError()
            if len(p1.questions) != len(p2.questions):
                raise ValueError()

            for q1, q2 in zip(p1.questions, p2.questions):
                if q1.words != q2.words or q1.question_id != q2.question_id:
                    raise ValueError()
                if [x.__dict__ for x in q1.answer.spans] != [x.__dict__ for x in q2.answer.spans]:
                    raise ValueError()


def main():
    stop = NltkPlusStopWords(True)
    prep = ExtractMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 3),
                                  intern=True, require_an_answer=True)
    train_batching = ClusteredBatcher(20, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(20, ContextLenKey(), False, False)
    builder = ConcatParagraphDatasetBuilder(train_batching, eval_batching, True)
    data = PreprocessedData(TriviaQaWebDataset(), prep, builder, eval_on_verified=False,
                            sample_dev=20, sample=100
                            )
    eval = [LossEvaluator(), BoundedSpanEvaluator([4, 8])]
    data.preprocess()

    for batch in list(data.get_train().get_epoch())[:10]:
        for point in batch:
            print(" ".join(point.question))
            print(point.answer.answer_text)
            context = point.get_context()
            for s,e in point.answer.answer_spans:
                print(context[s:e+1])


def show():
    stop = NltkPlusStopWords(True)
    prep = ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 3),
                                  WithIndicators(True, True), intern=True)
    train_batching = ClusteredBatcher(60, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, False)
    builder = ParagraphAndQuestionDatasetBuilder(train_batching, eval_batching)
    data = PreprocessedData(TriviaQaWebDataset(), prep, builder, eval_on_verified=False,
                            sample_dev=20, sample=100)
    data.preprocess(1)

    for batch in list(data.get_train().get_epoch())[:10]:
        for point in batch:
            print(" ".join(point.question))
            print(point.answer.answer_text)
            context = list(point.get_context())
            for s,e in point.answer.answer_spans:
                context[s] = "{{" + context[s]
                context[e] = context[e] + "}}"
            print(" ".join(context))
            input()


def main3():
    train_batching = ClusteredBatcher(10, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(10, ContextLenKey(), False, False)

    data = PreprocessedData(TriviaQaWebDataset(),
                            ExtractSingleParagraphFeaturized(MergeParagraphs(400), True, True),
                            ParagraphAndQuestionDatasetBuilder(train_batching, eval_batching),
                            sample_dev=10, sample=10,
                            eval_on_verified=False
                            )

    data.preprocess(1, 1000)
    train = data.get_train()
    for batch in train.get_epoch():
        for x in batch:
            print(x.question_id)
            print(x.question)
            print(" ".join("%s (%.3f)" % (w, f) for w, f in zip(x.question, x.q_features[:, 0])))
            print(" ".join("%s (%.3f)" % (w, f) for w, f in zip(x.get_context(), x.c_features[:, 0])))
            input()
            print(x.answer.answer_text)
            context = x.get_context()
            for s, e in x.answer.answer_spans:
                print(context[s:e+1])


def show():
    data = TriviaQaWebDataset()
    data = data.get_train()
    np.random.shuffle(data)
    for q in data[:500]:
        print(" ".join(q.question))



def check_tfidf():
    data = TriviaQaWebDataset()
    print("Loading questions")
    train = data.get_train()
    pairs = flatten_iterable([(q, d) for d in q.all_docs] for q in train)
    np.random.shuffle(pairs)
    check = pairs[:3000]
    sz = []
    split = MergeParagraphs(400)
    stop = NltkPlusStopWords(True)
    top = TopTfIdf(stop, 4)

    for q, d in tqdm(check):
        q_words = set(x.lower() for x in q.question)
        doc = data.evidence.get_document(d.doc_id)
        paras = split.split_annotated(doc, d.answer_spans)
        selected = top.prune(q.question, paras)


def check():
    data = TriviaQaWebDataset()
    train = data.get_train()
    pairs = flatten_iterable([(q, d.doc_id) for d in q.all_docs] for q in train)
    np.random.shuffle(pairs)
    check = pairs[:3000]
    sz = []
    split = MergeParagraphs(400)
    stop = NltkPlusStopWords(True).words
    r = re.compile("\w\w+")

    for q, d in tqdm(check):
        q_words = set(x.lower() for x in q.question)
        q_words = {x for x in q_words if r.fullmatch(x) is not None and x not in stop}
        doc = data.evidence.get_document(d)
        n_in_q = 0
        for para in split.split(doc):
            if any(word.lower() in q_words for word in flatten_iterable(para.text)):
                n_in_q += 1

        sz.append(n_in_q)

    sz = np.array(sz)
    c = Counter(sz)

    cum = 0
    for k in sorted(c.keys()):
        count = c[k]
        cum += count
        print("%d: %d %.4f %.4f" % (k, count, count/len(check), cum/len(check)))

    # train = {q.question_id: q for q in train}

    # raw = json.load(open(join(config.TRIVIA_QA, "qa", "web-dev.json")))["Data"]
    # for point in raw:
    #     id = point["QuestionId"]
    #     if id not in train:
    #         raise ValueError()
    #     ours = train[id]
    #     n_search = sum("Filename" in x for x in  point["SearchResults"])
    #     if n_search != len(ours.web_docs):
    #         raise ValueError()
    #     # n_entity = sum("Filename" in x for x in point["Entity"])


if __name__ == "__main__":
    show()