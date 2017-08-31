import argparse
import json

import numpy as np
from data_processing.qa_data import ParagraphAndQuestionDataset

import trainer
from dataset import FixedOrderBatcher
from evaluator import RecordQuestionId, RecordSpanPrediction, LossEvaluator, SpanEvaluator, SpanProbability
from experimental.niket_qa import NiketCorpus
from trainer import ModelDir
from utils import flatten_iterable, transpose_lists, print_table

"""
"""


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--official_output", type=str)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[4])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-s', '--step', type=int, default=None)
    parser.add_argument('-c', '--corpus', choices=["dev", "train", "test"], default="dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    corpus = NiketCorpus()
    if args.corpus == "dev":
        questions = corpus.get_dev()
    elif args.corpus == "train":
        questions = corpus.get_train()
    elif args.corpus == "test":
        questions = corpus.get_test()
    else:
        raise ValueError()

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]

    dataset = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluators = [LossEvaluator(), SpanEvaluator(args.answer_bounds), SpanProbability()]
    if args.official_output is not None:
        evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))
        evaluators.append(RecordQuestionId())

    if args.step is None:
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        checkpoint = model_dir.get_checkpoint(args.step)
    print(checkpoint)
    evaluation = trainer.test(model_dir.get_model(), evaluators, {args.corpus: dataset},
                              corpus.get_resource_loader(), checkpoint, args.ema)[args.corpus]

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))

    if args.official_output is not None:
        quid_to_context = {}
        for x in questions:
            quid_to_context[x.question_id] = flatten_iterable(x.context)

        q_id_to_answers = {}
        q_ids = evaluation.per_sample["question_id"]
        spans = evaluation.per_sample["bound-%d-span-predictions" % args.answer_bounds[0]]
        for q_id, (start, end) in zip(q_ids, spans):
            text = " ".join(quid_to_context[q_id][start:end+1])
            q_id_to_answers[q_id] = text

        with open(args.official_output, "w") as f:
            json.dump(q_id_to_answers, f)

if __name__ == "__main__":
    main()
    # tmp()




