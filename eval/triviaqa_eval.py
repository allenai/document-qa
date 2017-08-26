import argparse
import json
from typing import List

import numpy as np
from tqdm import tqdm

import trainer
from data_processing.document_splitter import TopTfIdf, MergeParagraphs
from data_processing.preprocessed_corpus import preprocess_par
from data_processing.qa_training_data import ContextLenKey, ParagraphAndQuestionDataset, ContextAndQuestion
from data_processing.text_utils import NltkPlusStopWords
from dataset import FixedOrderBatcher
from evaluator import Evaluator, Evaluation, RecordSpanPrediction, RecordQuestionId
from trainer import ModelDir
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.training_data import ExtractSingleParagraph
from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
from trivia_qa.triviaqa_evaluators import TfTriviaQaBoundedSpanEvaluator
from utils import ResourceLoader, flatten_iterable, transpose_lists, print_table

"""
Script to build an official trivia-qa prediction file
"""


class TriviaQaOracle(Evaluator):
    def tensors_needed(self, model):
        return {}

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kwargs):
        f1 = 0
        for point in data:
            answer = point.answer
            if answer is None:
                continue
            text = point.get_context()
            point_f1 = 0
            for s, e in answer.answer_spans:
                text = " ".join(text[s:e+1])
                for alias in answer.answer_text:
                    point_f1 = max(point_f1, trivia_f1_score(text, alias))
                    if point_f1 == 1:
                        break
                if point_f1 == 1:
                    break
            f1 += point_f1

        return Evaluation(dict(oracle_f1=f1/true_len))


def get_filename(file_id):
    return file_id[file_id.find("/")+1:]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--official_output", type=str)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('-a', '--async', type=int, default=None)
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[8])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["web-dev", "web-verified-dev"], default="web-dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    splitter = MergeParagraphs(800)
    para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), 1)
    prep = ExtractSingleParagraph(splitter, para_filter, False)

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

    pre = preprocess_par(trivia_questions, data.evidence, prep, 4, 1000)
    pre.data.sort(key=ContextLenKey())
    dataset = ParagraphAndQuestionDataset(pre.data, FixedOrderBatcher(args.batch_size, True),
                                          pre.true_len)

    evaluators = [TfTriviaQaBoundedSpanEvaluator(args.answer_bounds),
                  TriviaQaOracle()]
    if args.official_output is not None:
        evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))
        evaluators.append(RecordQuestionId())

    checkpoint = model_dir.get_latest_checkpoint()
    evaluation = trainer.test(model_dir.get_model(), evaluators, {args.corpus: dataset},
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
        for x in data:
            quid_to_context[x.question_id] = x.get_context()

        q_id_to_answers = {}
        q_ids = evaluation.per_sample["question_id"]
        spans = evaluation.per_sample["bound-%d-span-predictions" % args.answer_bounds[0]]
        for q_id, (start, end) in zip(q_ids, spans):
            text = " ".join(quid_to_context[q_id])[start:end+1]
            q_id_to_answers[q_id] = text

        with open(args.official_output, "w") as f:
            json.dump(q_id_to_answers, f)

if __name__ == "__main__":
    main()
    # tmp()




