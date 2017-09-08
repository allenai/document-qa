import numpy as np
from typing import List
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from configurable import Configurable
from data_processing.document_splitter import ParagraphFilter
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder
from data_processing.qa_training_data import ParagraphAndQuestionSpec, WordCounts, QaCorpusStats
from data_processing.span_data import TokenSpans
from dataset import Dataset, ListBatcher
from squad.squad_data import Document, Paragraph, DocParagraphAndQuestion
from utils import flatten_iterable


"""
Preprocessors to build a document-level question and answer dataset from SQuAD data
"""


class MultiParagraphSquadQuestion(object):
    def __init__(self, question_id, question, paragraphs: List[Paragraph],
                 paragraph_answer_spans, answer_text):
        self.question_id = question_id
        self.question = question
        self.paragraphs = paragraphs
        self.paragraph_answer_spans = paragraph_answer_spans
        self.answer_text = answer_text


class RandomSquadParagraphDataset(Dataset):
    def __init__(self,
                 questions: List[MultiParagraphSquadQuestion],
                 force_answer: float,
                 true_len: int,
                 batcher: ListBatcher):
        self.questions = questions
        self.force_answer = force_answer
        self.batcher = batcher
        self.true_len = true_len

    def get_vocab(self):
        voc = set()
        for q in self.questions:
            voc.update(q.question)
            for para in q.paragraphs:
                for sent in para.context:
                    voc.update(sent)
        return voc

    def get_spec(self):
        max_q_len = max(len(q.question) for q in self.questions)
        max_c_len = max(max(p.n_context_words for p in q.paragraphs) for q in self.questions)
        return ParagraphAndQuestionSpec(self.batcher.get_fixed_batch_size(), max_q_len,
                                        max_c_len, None)

    def get_samples(self, n_examples):
        n_batches = n_examples // self.batcher.batch_size
        return self.get_batches(n_batches), n_batches

    def get_epoch(self):
        questions = self.questions
        out = []
        for q in questions:
            if self.force_answer > 0 and np.random.random() < self.force_answer:
                candidates = [i for i,x in enumerate(q.paragraph_answer_spans) if len(x) > 0]
                if len(candidates) == 0:
                    raise ValueError()
                selected = candidates[np.random.randint(len(candidates))]
            else:
                selected = np.random.randint(len(q.paragraphs))

            para = q.paragraphs[selected]
            answer_spans = q.paragraph_answer_spans[selected]
            out.append(DocParagraphAndQuestion(q.question, TokenSpans(q.answer_text, answer_spans), q.question_id, para))

        return self.batcher.get_epoch(out)

    def percent_filtered(self):
        return (self.true_len - len(self.questions)) / self.true_len

    def __len__(self):
        return self.batcher.epoch_size(len(self.questions))


class RandomSquadParagraphBuilder(DatasetBuilder):
    def __init__(self, train_batching: ListBatcher, eval_batching: ListBatcher,
                 force_answer: float):
        self.train_batching = train_batching
        self.eval_batching = eval_batching
        self.force_answer = force_answer

    def build_stats(self, data: List[MultiParagraphSquadQuestion]):
        context_counts = Counter()
        question_counts = Counter()
        for point in data:
            question_counts.update(point.question)
            factor = 1/len(point.paragraphs)
            for para in point.paragraphs:
                for sent in para.context:
                    for word in sent:
                        context_counts[word] += factor
        return QaCorpusStats(question_counts, context_counts)

    def build_dataset(self, data, evidence, is_train: bool) -> Dataset:
        batching = self.train_batching if is_train else self.eval_batching
        return RandomSquadParagraphDataset(data, self.force_answer, len(data), batching)


class SquadTfIdfRanker(Preprocessor):
    def __init__(self, stop, n_to_select: int, force_answer: bool):
        self.stop = stop
        self.n_to_select = n_to_select
        self.force_answer = force_answer
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)

    def preprocess(self, question: List[Document], evidence):
        return self.ranked_questions(question)

    def rank(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        tfidf = self._tfidf
        para_features = tfidf.fit_transform([" ".join(" ".join(s) for s in x) for x in paragraphs])
        q_features = tfidf.transform([" ".join(q) for q in questions])
        scores = pairwise_distances(q_features, para_features, "cosine")
        return scores

    def ranked_questions(self, docs: List[Document]) -> List[MultiParagraphSquadQuestion]:
        out = []
        for doc in docs:
            scores = self.rank(flatten_iterable([q.words for q in x.questions] for x in doc.paragraphs),
                               [x.context for x in doc.paragraphs])
            q_ix = 0
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    para_scores = scores[q_ix]
                    para_ranks = np.argsort(para_scores)
                    selection = [i for i in para_ranks[:self.n_to_select]]
                    answer_spans = [np.zeros((0, 2), np.int32) for _ in selection]

                    if self.force_answer and para_ix not in selection:
                        selection[-1] = para_ix
                        answer_spans[-1] = q.answer.answer_spans
                    else:
                        correct_ix = [ix for ix,i in enumerate(selection) if i == para_ix]
                        if len(correct_ix) > 0:
                            answer_spans[correct_ix[0]] = q.answer.answer_spans

                    out.append(MultiParagraphSquadQuestion(q.question_id, q.words, [doc.paragraphs[i] for i in selection],
                                                           answer_spans, q.answer.answer_text))
                    q_ix += 1
        return out
