import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from docqa import trainer
from docqa.data_processing.qa_training_data import ContextAndQuestion, Answer, ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords, ParagraphWithInverse
from docqa.dataset import FixedOrderBatcher
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.evaluator import Evaluation, Evaluator
from docqa.model_dir import ModelDir
from docqa.squad.document_rd_corpus import get_doc_rd_doc
from docqa.squad.squad_data import SquadCorpus
from docqa.squad.squad_document_qa import SquadTfIdfRanker
from docqa.squad.squad_official_evaluation import exact_match_score as squad_em_score
from docqa.squad.squad_official_evaluation import f1_score as squad_f1_score
from docqa.utils import ResourceLoader, flatten_iterable, print_table


"""
Run an evaluation on "document-level" squad
"""


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

    def __init__(self, bound: int, record_text_ans: bool):
        self.bound = bound
        self.record_text_ans = record_text_ans

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        needed = dict(spans=span, model_scores=score)
        return needed

    def evaluate(self, data: List[RankedParagraphQuestion], true_len, **kargs):
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []

        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring"):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            pred_span = spans[i]
            pred_text = point.paragraph.get_original_text(pred_span[0], pred_span[1])
            if self.record_text_ans:
                text_answers.append(pred_text)
                if point.answer is None:
                    continue

            f1 = 0
            em = False
            for answer in data[i].answer.answer_text:
                f1 = max(f1, squad_f1_score(pred_text, answer))
                if not em:
                    em = squad_em_score(pred_text, answer)

            pred_f1s[i] = f1
            pred_em[i] = em

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        if self.record_text_ans:
            results["text_answer"] = text_answers
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["text_f1"] = pred_f1s
        results["rank"] = [x.rank for x in data]
        results["text_em"] = pred_em
        results["question_id"] = [x.question_id for x in data]
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on document-level SQuAD')
    parser.add_argument('model', help='model to use')
    parser.add_argument('output', type=str,
                        help="Store the per-paragraph results in csv format in this file")
    parser.add_argument('-n', '--n_sample', type=int, default=None,
                        help="(for testing) sample documents")
    parser.add_argument('-s', '--async', type=int, default=10,
                        help="Encoding batch asynchronously, queueing up to this many")
    parser.add_argument('-a', '--answer_bound', type=int, default=17,
                        help="Max answer span length")
    parser.add_argument('-p', '--n_paragraphs', type=int, default=None,
                        help="Max number of paragraphs to use")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-c', '--corpus', choices=["dev", "train", "doc-rd-dev"], default="dev")
    parser.add_argument('--no_ema', action="store_true",
                        help="Don't use EMA weights even if they exist")
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

        print("Fetching document reader docs...")
        doc_rd_versions = get_doc_rd_doc(docs)
        print("Ranking and matching with questions...")
        for doc in tqdm(docs):
            doc_questions = flatten_iterable(x.questions for x in doc.paragraphs)
            paragraphs = doc_rd_versions[doc.title]
            ranks = ranker.rank([x.words for x in doc_questions], [x.text for x in paragraphs])
            for i, question in enumerate(doc_questions):
                para_ranks = np.argsort(ranks[i])
                for para_rank, para_num in enumerate(para_ranks[:args.n_paragraphs]):
                    # Just use dummy answers spans for these pairs
                    questions.append(RankedParagraphQuestion(question.words,
                                            TokenSpans(question.answer.answer_text, np.zeros((0, 2), dtype=np.int32)),
                                            question.question_id, paragraphs[para_num], para_rank, para_num))
        rl = ResourceLoader()
    else:
        if args.corpus == "dev":
            docs = SquadCorpus().get_dev()
        else:
            docs = SquadCorpus().get_train()
        rl = SquadCorpus().get_resource_loader()

        if args.n_sample is not None:
            docs.sort(key=lambda x:x.doc_id)
            np.random.RandomState(0).shuffle(docs)
            docs = docs[:args.n_sample]

        for q in ranker.ranked_questions(docs):
            for i, p in enumerate(q.paragraphs):
                questions.append(RankedParagraphQuestion(q.question,
                                                         TokenSpans(q.answer_text, p.answer_spans),
                                                         q.question_id,
                                                         ParagraphWithInverse([p.text], p.original_text, p.spans),
                                                         i, p.paragraph_num))

    print("Split %d docs into %d paragraphs" % (len(docs), len(questions)))

    questions = sorted(questions, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)
    for q in questions:
        if len(q.answer.answer_spans.shape) != 2:
            raise ValueError()

    checkpoint = model_dir.get_best_weights()
    if checkpoint is not None:
        print("Using best weights")
    else:
        print("Using latest checkpoint")
        checkpoint = model_dir.get_latest_checkpoint()
        if checkpoint is None:
            raise ValueError("No checkpoints found")

    data = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    model = model_dir.get_model()
    evaluation = trainer.test(model, [RecordParagraphSpanPrediction(args.answer_bound, True)],
                              {args.corpus: data}, rl, checkpoint,
                              not args.no_ema, args.async)[args.corpus]

    print("Saving result")
    output_file = args.output

    df = pd.DataFrame(evaluation.per_sample)

    df.sort_values(["question_id", "rank"], inplace=True, ascending=True)
    group_by = ["question_id"]
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    table = [["N Paragraphs", "EM", "F1"]]
    table += list([str(i+1), "%.4f" % e, "%.4f" % f] for i, (e, f) in enumerate(zip(em, f1)))
    print_table(table)

    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()




