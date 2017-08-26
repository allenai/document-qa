import argparse
import json

import numpy as np

import trainer
from data_processing.qa_training_data import ParagraphAndQuestionDataset
from dataset import FixedOrderBatcher
from evaluator import RecordQuestionId, RecordSpanPrediction
from squad.squad_data import SquadCorpus, split_docs
from squad.squad_eval import BoundedSquadSpanEvaluator
from trainer import ModelDir
from utils import flatten_iterable, transpose_lists, print_table

"""
"""


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--official_output", type=str)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[17])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    corpus = SquadCorpus()
    if args.corpus == "dev":
        questions = corpus.get_dev()
    else:
        questions = corpus.get_train()
    questions = split_docs(questions)

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]

    dataset = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluators = [BoundedSquadSpanEvaluator(args.answer_bounds)]
    if args.official_output is not None:
        evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))
        evaluators.append(RecordQuestionId())

    checkpoint = model_dir.get_latest_checkpoint()
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
        quid_to_para = {}
        for x in questions:
            quid_to_para[x.question_id] = x.paragraph

        q_id_to_answers = {}
        q_ids = evaluation.per_sample["question_id"]
        spans = evaluation.per_sample["bound-%d-span-predictions" % args.answer_bounds[0]]
        for q_id, (start, end) in zip(q_ids, spans):
            text = quid_to_para[q_id].get_original_text(start, end)
            q_id_to_answers[q_id] = text

        with open(args.official_output, "w") as f:
            json.dump(q_id_to_answers, f)

if __name__ == "__main__":
    main()
    # tmp()




