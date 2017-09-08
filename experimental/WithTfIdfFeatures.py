import numpy as np
from typing import List

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from data_processing.document_splitter import DocumentSplitter
from data_processing.preprocessed_corpus import Preprocessor, FilteredData
from data_processing.qa_training_data import Answer, ParagraphAndQuestion
from data_processing.span_data import TokenSpans
from data_processing.text_utils import NltkPlusStopWords
from trivia_qa.read_data import TriviaQaQuestion
from trivia_qa.training_data import DocumentParagraphQuestion
from utils import flatten_iterable


class FeaturizedParagraphAndQuestion(ParagraphAndQuestion):
    def __init__(self, q_id: str, doc_id: str, para_range, question: List[str],
                 context: List[str], answer: Answer, c_features, q_features):
        super().__init__(context, question, answer, q_id)
        self.doc_id = doc_id
        self.para_range = para_range
        self.c_features = c_features
        self.q_features = q_features


class ExtractSingleParagraphFeaturized(Preprocessor):
    def __init__(self, splitter: DocumentSplitter, intern, require_answer=True):
        self.splitter = splitter
        self.intern = intern
        self.require_answer = require_answer
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=NltkPlusStopWords(True).words)

    def n_features(self):
        return 1

    def preprocess(self, questions: List[TriviaQaQuestion], evidence) -> FilteredData:
        splitter = self.splitter
        output = []
        read_only = splitter.reads_first_n
        for q in questions:
            for doc in q.all_docs:
                text = evidence.get_document(doc.doc_id, n_tokens=read_only)
                if text is None:
                    raise ValueError(doc.doc_id, doc.doc_id)

                paragraphs = splitter.split_annotated(text, doc.answer_spans)

                tfidf = self._tfidf
                text = []
                for para in paragraphs:
                    text.append(" ".join(" ".join(s) for s in para.text))
                try:
                    para_features = tfidf.fit_transform(text)
                    q_features = tfidf.transform([" ".join(q.question)])
                except ValueError:
                    continue

                fns = {w: i for i, w in enumerate(tfidf.get_feature_names())}

                dists = pairwise_distances(q_features, para_features, "cosine").ravel()
                ix = np.argmin(dists)
                pre = tfidf.build_preprocessor()
                if dists[ix] == 0:
                    continue
                selected = paragraphs[ix]
                if self.require_answer and len(selected.answer_spans) == 0:
                    continue

                text = flatten_iterable(selected.text)
                para_features = para_features.todok()  # for fast lookup
                features = np.zeros(len(text), dtype=np.float32)
                for i, word in enumerate(text):
                    word_ix = fns.get(pre(word))
                    if word_ix is not None:
                        features[i] = para_features[ix, word_ix]
                features = np.expand_dims(features, 1)

                question_features = np.zeros(len(q.question), dtype=np.float32)
                q_features = q_features.todok()
                for i, word in enumerate(q.question):
                    word_ix = fns.get(pre(word))
                    if word_ix is not None:
                        question_features[i] = q_features[0, word_ix]

                output.append(FeaturizedParagraphAndQuestion(
                    q.question_id, doc.doc_id, (selected.start, selected.end),
                    q.question, text, TokenSpans(q.answer.all_answers, selected.answer_spans),
                    features, np.expand_dims(question_features, 1)))
        return FilteredData(output, sum(len(x.all_docs) for x in questions))

    def finalize(self, x: FilteredData):
        if self.intern:
            question_map = {}
            for q in x.data:
                q.question_id = sys.intern(q.question_id)
                if q.question_id in question_map:
                    q.question = question_map[q.question_id]
                else:
                    q.question = tuple(sys.intern(w) for w in q.question)
                    question_map[q.question_id] = q.question
                q.doc_id = sys.intern(q.doc_id)
                q.context = [sys.intern(w) for w in q.context]
