import numpy as np
from os.path import join

import config
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.multi_paragraph_qa import ConcatParagraphDatasetBuilder, RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from evaluator import LossEvaluator
from squad.squad_data import SquadCorpus
from trivia_qa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWebDataset
from trivia_qa.training_data import ExtractMultiParagraphs
from trivia_qa.triviaqa_evaluators import BoundedSpanEvaluator

import tensorflow as tf


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


def main3():
    stop = NltkPlusStopWords(True)
    prep = ExtractMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 4),
                                  intern=True, require_an_answer=True)
    builder = RandomParagraphSetDatasetBuilder(10, 10, 2, True)
    data = PreprocessedData(TriviaQaWebDataset(), prep, builder, eval_on_verified=False,
                            sample_dev=10, sample=20
                            )
    data.preprocess(2, 100)
    for batch in data.get_train().get_epoch():
        for x in batch:
            print(x.question_id)
            print(x.question)
            print(x.answer.answer_textw)
            context = x.get_context()
            for s, e in x.answer.answer_spans:
                print(context[s:e+1])


if __name__ == "__main__":
    main1()