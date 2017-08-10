import pickle
from collections import Counter
from os import makedirs, listdir
from os.path import isfile, join, exists, isdir
from typing import List, Tuple, Optional

import numpy as np

from config import CORPUS_DIR
from configurable import Configurable
from data_processing.qa_data import ParagraphAndQuestionSpec, Answer, ParagraphAndQuestion, QaCorpus, \
    QaCorpusStats
from data_processing.word_vectors import load_word_vectors
from utils import ResourceLoader

"""
Classes for representing a QA dataset that has questions per a paragraph,
and those paragraphs are organized into documents
"""


class Question(object):
    """ Question paired with its answer """

    def __init__(self, question: List[str], answer: Answer, question_id: str):
        self.words = question
        self.question_id = question_id
        self.answer = answer

    def __repr__(self) -> str:
        return " ".join(self.words)


class Paragraph(object):
    """ Context with multiple questions """

    def __init__(self,
                 context: List[List[str]],
                 questions: List[Question],
                 article_id: str,
                 paragraph_num: int,
                 original_text: Optional[str] = None,
                 spans: Optional[np.ndarray] = None):
        self.article_id = article_id
        self.context = context
        self.questions = questions
        self.paragraph_num = paragraph_num
        self.original_text = original_text
        self.spans = spans

    def get_original_text(self, start, end):
        return self.original_text[self.spans[start][0]:self.spans[end][1]]

    @property
    def n_context_words(self):
        return sum(len(x) for x in self.context)

    def __repr__(self) -> str:
        return "Paragraph%d(%s...)" % (self.paragraph_num, self.context[0][:40])


class Document(object):
    """ Collection of paragraphs """

    def __init__(self, doc_id: str, title: str,
                 wiki_title: Optional[str], paragraphs: List[Paragraph]):
        self.wiki_title = wiki_title
        self.title = title
        self.doc_id = doc_id
        self.paragraphs = paragraphs

    def __repr__(self) -> str:
        return "Document(%s)" % self.title


class ContextLenKey(Configurable):
    def __call__(self, q: ParagraphAndQuestion):
        return sum(len(s) for s in q.context)


class ContextLenBucketedKey(Configurable):
    def __init__(self, bucket_size: int):
        self.bucket_size = bucket_size

    def __call__(self, q: ParagraphAndQuestion):
        return sum(len(s) for s in q.context)//self.bucket_size


class DocParagraphAndQuestion(object):
    """ QA pair the comes from a specific document/paragraph, and optionally mantains a
     "reverse" mapping from its tokenized form to its original text """

    def __init__(self, question: List[str], answer: Optional[Answer], question_id: str, paragraph: Paragraph):
        self.question = question
        self.answer = answer
        self.question_id = question_id
        self.paragraph = paragraph

    def get_original_text(self, para_start, para_end):
        return self.paragraph.get_original_text(para_start, para_end)

    @property
    def context(self):
        return self.paragraph.context

    @property
    def paragraph_num(self):
        return self.paragraph.paragraph_num

    @property
    def article_id(self):
        return self.paragraph.article_id


class ParagraphQaStats(object):
    def __init__(self, docs: List[Document], special_tokens=None):
        self.docs = docs
        self._question_counts = None
        self._context_counts = None
        self._unique_context_counts = None
        self.special_tokens = special_tokens

    def _load(self):
        if self._context_counts is not None:
            return
        self._context_counts = Counter()
        self._unique_counts = Counter()
        self._question_counts = Counter()
        for doc in self.docs:
            for para in doc.paragraphs:
                for sent in para.context:
                    self._unique_counts.update(sent)
                    for word in sent:
                        self._context_counts[word] += len(para.questions)
                for question in para.questions:
                    self._question_counts.update(question.words)
                    self._question_counts.update(question.answer.get_vocab())

        return self._question_counts

    def get_question_counts(self):
        self._load()
        return self._question_counts

    def get_context_counts(self):
        self._load()
        return self._context_counts

    def get_document_counts(self):
        self._load()
        return self._unique_context_counts

    def get_word_counts(self):
        return self.get_context_counts() + self.get_question_counts()


def split_docs(docs: List[Document]) -> List[DocParagraphAndQuestion]:
    paras = []
    for doc in docs:
        for i, para in enumerate(doc.paragraphs):
            for question in para.questions:
                paras.append(DocParagraphAndQuestion(question.words, question.answer, question.question_id, para))
    return paras


def build_qa_collection(docs: List[Document]):
    context = Counter()
    unique_counts = Counter()
    question_counts = Counter()
    for doc in docs:
        for para in doc.paragraphs:
            for sent in para.context:
                unique_counts.update(sent)
                for word in sent:
                    context[word] += len(para.questions)
            for question in para.questions:
                question_counts.update(question.words)
                question_counts.update(question.answer.get_vocab())
    return QaCorpusStats(question_counts, context, unique_counts)


# TODO we should go back to making loading the reverse-mappings optional
class DocumentCorpus(QaCorpus):
    TRAIN_FILE = "train.pkl"
    TEST_FILE = "test.pkl"
    DEV_FILE = "dev.pkl"

    VOCAB_FILE = "vocab.txt"
    WORD_VEC_SUFFIX = "_pruned"

    @staticmethod
    def make_corpus(dir, train: List[Document],
                    dev: List[Document],
                    test: List[Document]):
        if isfile(dir) or (exists(dir) and len(listdir(dir))) > 0:
            raise ValueError()
        if not exists(dir):
            makedirs(dir)

        for name, data in [(DocumentCorpus.TRAIN_FILE, train), (DocumentCorpus.DEV_FILE, dev), (DocumentCorpus.TEST_FILE, test)]:
            if data is not None:
                with open(join(dir, name), 'wb') as f:
                    pickle.dump(data, f)

    def __init__(self, source_name):
        self.source_name = source_name
        dir = join(CORPUS_DIR, source_name)
        if not exists(dir):
            raise ValueError("No directory: " + dir)
        if not isdir(dir):
            raise ValueError("Not a directory: " + dir)
        self.dir = dir

    def get_vocab(self):
        """ get all-lower cased unique words for this corpus, includes train/dev/test files """
        voc_file = join(self.dir, self.VOCAB_FILE)
        if exists(voc_file):
            with open(voc_file, "r") as f:
                return [x.rstrip() for x in f]
        else:
            voc = set()
            for fn in [self.get_train_docs, self.get_dev_docs, self.get_test_docs]:
                for doc in fn():
                    for para in doc.paragraphs:
                        for sent in para.context:
                            voc.update(x.lower() for x in sent)
                        for question in para.questions:
                            voc.update(x.lower() for x in question.words)
                            voc.update(x.lower() for x in question.answer.get_vocab())
            voc_list = sorted(list(voc))
            with open(voc_file, "w") as f:
                for word in voc_list:
                    f.write(word)
                    f.write("\n")
            return voc_list

    def get_pruned_word_vecs(self, word_vec_name, voc=None):
        """
        Loads word vectors that have been pruned to the case-insensitive vocab of this corpus.
        WARNING: this includes test/dev words

        This exists since loading word-vecs each time we startup can be a big pain, so
        we cache the pruned vecs on-disk as a .npy file we can re-load quickly.
        """

        vec_file = join(self.dir, word_vec_name + self.WORD_VEC_SUFFIX + ".npy")
        if isfile(vec_file):
            print("Loading word vec %s for %s from cache" % (word_vec_name, self.name))
            with open(vec_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Building pruned word vec %s for %s" % (self.name, word_vec_name))
            voc = self.get_vocab()
            vecs = load_word_vectors(word_vec_name, voc)
            with open(vec_file, "wb") as f:
                pickle.dump(vecs, f)
            return vecs

    def get_resource_loader(self):
        return ResourceLoader(self.get_pruned_word_vecs)

    def get_train(self) -> List[DocParagraphAndQuestion]:
        return split_docs(self.get_train_docs())

    def get_train_corpus(self) -> Tuple[List[ParagraphAndQuestion], ParagraphQaStats]:
        docs = self.get_train_docs()
        collection = ParagraphQaStats(docs)
        return split_docs(docs), collection

    def get_dev(self) -> List[DocParagraphAndQuestion]:
        return split_docs(self.get_dev_docs())

    def get_test(self) -> List[DocParagraphAndQuestion]:
        return split_docs(self.get_test_docs())

    def get_train_docs(self) -> List[Document]:
        return self._load(join(self.dir, self.TRAIN_FILE))

    def get_dev_docs(self) -> List[Document]:
        return self._load(join(self.dir, self.DEV_FILE))

    def get_test_docs(self) -> List[Document]:
        return self._load(join(self.dir, self.TEST_FILE))

    def _load(self, file) -> List[Document]:
        if not exists(file):
            return []
        with open(file, "rb") as f:
            return pickle.load(f)

    def _read_answer(self, json_obj) -> Answer:
        # subclasses need to implement this for their answer format
        raise NotImplemented()

    @property
    def name(self):
        return self.source_name

    def __setstate__(self, state):
        self.source_name = state["state"]["source_name"]
        dir = join(CORPUS_DIR, self.source_name)
        if not exists(dir):
            raise ValueError("No directory: " + dir)
        if not isdir(dir):
            raise ValueError("Not a directory: " + dir)
        self.dir = dir


def compute_document_voc(data: List[Document]):
    voc = set()
    for doc in data:
        for para in doc.paragraphs:
            for sent in para.context:
                voc.update(sent)
            for question in para.questions:
                voc.update(question.words)
                voc.update(question.answer.get_vocab())
    return voc


def get_doc_input_spec(batch_size, data: List[List[Document]]) -> ParagraphAndQuestionSpec:
    max_num_sents = 0
    max_sent_size = 0
    max_ques_size = 0
    max_word_size = 0
    max_para_size = 0
    for docs in data:
        for doc in docs:
            for para in doc.paragraphs:
                max_num_sents = max(max_num_sents, len(para.context))
                max_sent_size = max(max_sent_size, max(len(s) for s in para.context))
                max_word_size = max(max_word_size, max(len(word) for sent in para.context for word in sent))
                max_para_size = max(max_para_size, sum(len(sent) for sent in para.context))
                for question in para.questions:
                    max_ques_size = max(max_ques_size, len(question.words))
                    max_word_size = max(max_word_size, max(len(word) for word in question.words))
    return ParagraphAndQuestionSpec(batch_size, max_ques_size, max_para_size, max_num_sents, max_sent_size, max_word_size)

