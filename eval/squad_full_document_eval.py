import argparse
import json
import pickle
from os.path import join
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import trainer
from data_processing.document_splitter import DocumentSplitter, MergeParagraphs, ParagraphFilter, ContainsQuestionWord, \
    TopTfIdf

from data_processing.multi_paragraph_qa import DocumentParagraph
from data_processing.qa_training_data import ContextAndQuestion, Answer, ParagraphAndQuestionDataset
from data_processing.span_data import TokenSpans
from data_processing.text_utils import NltkPlusStopWords, ParagraphWithInverse
from dataset import FixedOrderBatcher
from evaluator import Evaluator, Evaluation
from squad.document_rd_corpus import get_doc_rd_doc
from squad.squad_document_qa import SquadTfIdfRanker
from squad.squad_data import SquadCorpus, Paragraph, DocParagraphAndQuestion
from trainer import ModelDir
from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
from utils import ResourceLoader, flatten_iterable


class RankedParagraphQuestion(ContextAndQuestion):

    def __init__(self, question: List[str], answer: Optional[Answer],
                 question_id: str, paragraph: ParagraphWithInverse,
                 rank: int, paragraph_number: int):
        super().__init__(question, answer, question_id)
        self.paragraph = paragraph
        self.rank = rank
        self.paragraph_number = paragraph_number

    def get_original_text(self, para_start, para_end):
        return self.paragraph.get_original_text(para_start, para_end)

    def get_context(self):
        return flatten_iterable(self.paragraph.text)

    @property
    def n_context_words(self) -> int:
        return sum(len(s) for s in self.paragraph.text)


class RecordParagraphSpanPrediction(Evaluator):
    """ Computes the best span in tensorflow, meaning which we expect to be faster and
    does not require us having to keep the entire set of output logits in RAM """
    def __init__(self, bound: int):
        self.bound = bound

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[RankedParagraphQuestion], true_len, **kargs):
        print("Begining evaluation")
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        oracle_f1s = np.zeros(len(data))

        print(" Scoring...")
        for i in tqdm(range(len(data)), total=len(data), ncols=80):
            point = data[i]
            if point.answer is None:
                continue
            pred_span = spans[i]
            pred_text = point.paragraph.get_original_text(pred_span[0], pred_span[1])
            f1 = 0
            for answer in data[i].answer.answer_text:
                f1 = max(f1, trivia_f1_score(pred_text, answer))
            pred_f1s[i] = f1

            oracle_f1 = 0
            for s, e in point.answer.answer_spans:
                pred_text = point.paragraph.get_original_text(s, e)
                for answer in point.answer.answer_text:
                    oracle_f1 = max(oracle_f1, trivia_f1_score(pred_text, answer))
            oracle_f1s[i] = oracle_f1

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["rank"] = [x.rank for x in data]
        results["text_f1"] = pred_f1s
        results["oracle_f1"] = oracle_f1s
        results["para_number"] = np.array([x.paragraph_number for x in data])
        results["question_id"] = [x.question_id for x in data]
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument('output', type=str)
    parser.add_argument('-n', '--n_sample', type=int, default=None)
    parser.add_argument('-s', '--async', type=int, default=10)
    parser.add_argument('-a', '--answer_bound', type=int, default=17)
    parser.add_argument('-p', '--n_paragraphs', type=int, default=None)
    parser.add_argument('-r', '--rank', type=str, default="none", choices=["tfidf", "none"])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["dev", "train", "doc-rd-dev"], default="dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    print("Loading data")

    questions = []
    ranker = SquadTfIdfRanker(NltkPlusStopWords(True),
                              args.n_paragraphs, force_answer=False)

    if args.corpus == "doc-rd-dev":
        docs = SquadCorpus().get_dev()
        if args.n_sample is not None:
            docs.sort(key=lambda x:x.doc_id)
            np.random.RandomState(0).shuffle(docs)
            docs = docs[:args.n_sample]

        doc_rd_versions = get_doc_rd_doc(docs)
        for doc in docs:
            questions = flatten_iterable(x.questions for x in doc.paragraphs)
            paragraphs = doc_rd_versions[doc.title]
            ranks = ranker.rank(questions, [x.text for x in paragraphs])
            for i, question in enumerate(questions):
                para_ranks = np.argsort(ranks[i])
                for para_rank, para_num in enumerate(para_ranks[:args.n_paragraphs]):
                    RankedParagraphQuestion(question.words,
                                            TokenSpans(question.answer_text, np.zeros((0, 2))),
                                            question.question_id, paragraphs[para_num], para_rank)
    else:
        if args.corpus == "dev":
            docs = SquadCorpus().get_dev()
        else:
            docs = SquadCorpus().get_train()

        if args.n_sample is not None:
            docs.sort(key=lambda x:x.doc_id)
            np.random.RandomState(0).shuffle(docs)
            docs = docs[:args.n_sample]

        for q in ranker.ranked_questions(docs):
            for i, p in enumerate(q.paragraphs):
                questions.append(RankedParagraphQuestion(q.question,
                                                         TokenSpans(q.answer_text, q.paragraph_answer_spans[i]),
                                                         q.question_id,
                                                         ParagraphWithInverse(p.context, p.original_text, p.spans),
                                                         i, p.paragraph_num))

    print("Split %d docs into %d paragraphs" % (len(docs), len(questions)))

    questions = sorted(questions, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    checkpoint = model_dir.get_latest_checkpoint()
    data = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    model = model_dir.get_model()
    evaluation = trainer.test(model, [RecordParagraphSpanPrediction(args.answer_bound)],
                              {args.corpus: data}, ResourceLoader(), checkpoint,
                              args.ema, args.async)[args.corpus]

    print("Saving result")
    # output_file = join(model_dir.get_eval_dir(), args.output)
    output_file = args.output

    if output_file.endswith("json"):
        with open(output_file, "w") as f:
            json.dump(evaluation.per_sample, f)
    elif output_file.endswith("pkl"):
        with open(output_file, "wb") as f:
            pickle.dump(evaluation.per_sample, f)
    elif output_file.endswith("csv"):
        import pandas as pd
        df = pd.DataFrame(evaluation.per_sample)
        df.to_csv(output_file, index=False)
    else:
        raise ValueError("Unrecognized file format")

if __name__ == "__main__":
    main()
    # tmp()




