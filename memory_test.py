import os
import resource

import sys

import gc

import time
from tqdm import tqdm

from data_processing.document_splitter import MergeParagraphs, TopTfIdf, First
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import ContextLenBucketedKey, ContextLenKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.multi_paragraph_data import ExtraMultiParagraphs, MultiParagraphDatasetBuilder
from utils import flatten_iterable


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    return psutil.virtual_memory()


def test_load():
    data = TriviaQaWebDataset()
    print("Loading questions..")
    questions = data.get_train()
    evidence = data.evidence

    docs = []
    for i, q in enumerate(questions):
        for doc in q.all_docs:
            text = flatten_iterable(evidence.get_document(doc.doc_id, 400*4))
            # for j, sent in enumerate(text):
            #     text[j] = [sys.intern(x) for x in sent]
            text = tuple(sys.intern(x) for x in flatten_iterable(text))
            docs.append(text)
        if i % 1000 == 0:
            print("%d: %.4f" % (i, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024)))


def test_extract():
    stop = NltkPlusStopWords()
    train_batching = ClusteredBatcher(60, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, False)
    data = PreprocessedData(TriviaQaWebDataset(),
                            ExtraMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 4), intern=True, require_an_answer=True),
                            MultiParagraphDatasetBuilder(train_batching, eval_batching, 0.5),
                            sample=1000, sample_dev=1000,
                            eval_on_verified=False
                            )
    data.preprocess(6, 1000)
    data.cache_preprocess("tmp.pkl.gz")
    # t0 = time.perf_counter()
    data.load_preprocess("tmp.pkl.gz")
    for point in data.get_train().get_epoch():
        pass
    # print(time.perf_counter() - t0)
    # print(memory_usage_psutil())
    # for q in (data._train.questions + data._dev.questions):
    #     q.question_id = sys.intern(q.question_id)
    #     q.question = [sys.intern(x) for x in q.question]
    #     q.paragraphs = None
        # for para in q.paragraphs:
            # para.text = flatten_iterable(para.text)
            # para.text = None

    # data._train = None
    # data._dev = None
    # gc.collect()
    # print(memory_usage_psutil())

    # tr = data.get_train()
    # print(sum(len(x) for x in tr.get_epoch()))



if __name__ == "__main__":
    test_extract()