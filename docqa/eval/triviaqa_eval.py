import argparse
import json
from typing import List

import numpy as np
from docqa.data_processing.document_splitter import TopTfIdf, MergeParagraphs, Truncate
from docqa.data_processing.qa_training_data import ContextLenKey, ParagraphAndQuestionDataset, ContextAndQuestion
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import FixedOrderBatcher
from docqa.evaluator import Evaluator, Evaluation, SpanEvaluator
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset
from docqa.triviaqa.training_data import ExtractSingleParagraph
from docqa.utils import ResourceLoader, transpose_lists, print_table

from docqa import trainer
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.model_dir import ModelDir

"""
Script to build an official trivia-qa prediction file
"""


class RecordSpanPrediction(Evaluator):
    def __init__(self, bound: int):
        self.bound = bound

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        spans, model_scores = kargs["spans"], kargs["model_scores"]
        results = {"model_conf": model_scores,
                   "predicted_span": spans,
                   "question_id": [x.question_id for x in data],
                   "doc_id": [x.doc_id for x in data]}
        return Evaluation({}, results)


class HasAnswerSpan(Evaluator):
    def tensors_needed(self, model):
        return {}

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kwargs):
        return Evaluation(dict(has_answer=sum(len(x.answer.answer_spans) > 0 for x in data)/true_len))


def get_filename(file_id):
    return file_id[file_id.find("/")+1:]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--official_output", type=str)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('-s', '--selector', default="tfidf")
    parser.add_argument('-t', '--n_tokens', default=800, type=int)
    parser.add_argument('-a', '--async', type=int, default=None)
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[8])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["web-dev", "web-verified-dev"], default="web-dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    model = model_dir.get_model()

    if args.selector == "tfidf":
        splitter = MergeParagraphs(args.n_tokens)
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), 1)
    elif args.selector == "truncate":
        splitter = Truncate(args.n_tokens)
        para_filter = None
    else:
        raise ValueError("Unknown selector: " + args.selector)

    prep = ExtractSingleParagraph(splitter, para_filter, model.preprocessor, False, False)

    if args.corpus == "web-dev" or args.corpus == "web-verified-dev":
        data = TriviaQaWebDataset()
        print("Loading questions...")
        if args.corpus == "web-dev":
            trivia_questions = data.get_dev()
        else:
            trivia_questions = data.get_verified()
    else:
        raise RuntimeError()

    if args.sample_questions:
        trivia_questions.sort(key=lambda x: x.question_id)
        trivia_questions = np.random.RandomState(0).choice(trivia_questions, args.sample_questions, replace=False)

    pre = preprocess_par(trivia_questions, data.evidence, prep, 6, 1000)
    pre.data.sort(key=ContextLenKey())

    dataset = ParagraphAndQuestionDataset(pre.data, FixedOrderBatcher(args.batch_size, True), pre.true_len)

    evaluators = [HasAnswerSpan(), SpanEvaluator(args.answer_bounds, text_eval="triviaqa")]

    if args.official_output is not None:
        evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))

    checkpoint = model_dir.get_latest_checkpoint()

    evaluation = trainer.test(model, evaluators, {args.corpus: dataset},
                              ResourceLoader(), checkpoint, args.ema, args.async)[args.corpus]

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))

    if args.official_output is not None:
        quid_to_context = {}
        for x in pre.data:
            quid_to_context[(x.question_id, x.doc_id)] = x.get_context()

        q_id_to_answers = {}
        q_ids = evaluation.per_sample["question_id"]
        doc_ids = evaluation.per_sample["doc_id"]
        spans = evaluation.per_sample["predicted_span"]
        for q_id, doc_id, (start, end) in zip(q_ids, doc_ids, spans):
            file_name = data.evidence.file_id_map[doc_id]
            if file_name.startswith("web/"):
                file_name = file_name[4:]
            elif file_name.startswith("wikipedia/"):
                file_name = file_name[10:]
            else:
                raise ValueError()
            file_name += ".txt"

            text = " ".join(quid_to_context[(q_id, doc_id)][start:end+1])
            q_id_to_answers[q_id + "--" + file_name] = text

        with open(args.official_output, "w") as f:
            json.dump(q_id_to_answers, f)

if __name__ == "__main__":
    main()
    # tmp()




