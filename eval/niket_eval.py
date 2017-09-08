import argparse
import json
from typing import List

import numpy as np

import trainer
from data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextAndQuestion
from data_processing.span_data import span_f1
from dataset import FixedOrderBatcher
from evaluator import RecordQuestionId, RecordSpanPrediction, LossEvaluator, SpanEvaluator, SpanProbability, Evaluator, \
    Evaluation, span_scores
from experimental.niket_qa.build_niket_dataset import NiketCorpus
from squad.squad_official_evaluation import f1_score
from trainer import ModelDir
from utils import flatten_iterable, transpose_lists, print_table

"""
"""


class RecordSpanPredictionNiket(Evaluator):
    def __init__(self, bound: int):
        self.bound = bound
        print("INIT")

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score,
                    none_score=prediction.none_logit,
                    none_prob=prediction.none_prob)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        spans, model_scores = kargs["spans"], kargs["model_scores"]
        scores = span_scores(data, spans)
        has_answer = [len(x.answer.answer_spans) > 0 for x in data]
        results = {"model_scores": model_scores,
                   "spans": spans,
                   "has_answer": has_answer,
                   "f1": scores[:, 0],
                   "accuracy": scores[:, 1],
                   "none_prob": np.array(kargs["none_prob"]),
                   "none_score": np.array(kargs["none_score"]),
                   "question_id": [x.question_id for x in data]
                   }
        return Evaluation({}, results)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--official_output", type=str)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[4])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-s', '--step', type=int, default=None)
    parser.add_argument('-c', '--corpus', choices=["dev", "train", "test"], default=["dev"], nargs="+")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    quid_to_q = {}

    corpus = NiketCorpus()
    data = {}
    for c in args.corpus:
        if c == "dev":
            questions = corpus.get_dev()
        elif c == "train":
            questions = corpus.get_train()
        elif c == "test":
            questions = corpus.get_test()
        else:
            raise ValueError()

        if args.sample_questions:
            np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
            questions = questions[:args.sample_questions]

        for x in questions:
            quid_to_q[x.question_id] = x

        dataset = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))
        data[c] = dataset

    evaluators = [LossEvaluator()]
    if args.official_output is not None:
        evaluators.append(RecordSpanPredictionNiket(args.answer_bounds[0]))

    if args.step is None:
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        checkpoint = model_dir.get_checkpoint(args.step)

    evaluation = trainer.test(model_dir.get_model(), evaluators, data,
                              corpus.get_resource_loader(), checkpoint, args.ema)

    print("Choosing threshold")
    e = evaluation["dev"].per_sample
    q_ids = e["question_id"]
    spans = e["spans"]
    confs = e["none_prob"]
    for th in [0, 0.1, 0.15, 0.2, 0.25]:
        score = 0
        none = 0
        for q_id, (start, end), conf in zip(q_ids, spans, confs):
            answer = quid_to_q[q_id].answer
            if conf < th:
                text = " ".join(quid_to_q[q_id].get_context()[start:end + 1])
                if len(answer.answer_text) > 0:
                    score += max(f1_score(a, text) for a in answer.answer_text)
            else:
                none += 1
                if len(answer.answer_text) == 0:
                    score += 1
        print("%s: %.4f (predicted %d (%.4f))" % (str(th), score/len(q_ids), none, none/len(q_ids)))

    # Print the scalar results in a two column table
    for name, evaluation in evaluation.items():
        scalars = evaluation.scalars
        cols = list(sorted(scalars.keys()))
        table = [cols]
        header = ["Metric", ""]
        table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
        print_table([header] + transpose_lists(table))

        if args.official_output is not None:
            q_id_to_answers = {}
            q_ids = evaluation.per_sample["question_id"]
            spans = evaluation.per_sample["spans"]
            confs = evaluation.per_sample["none_prob"]
            score = 0
            for q_id, (start, end), conf in zip(q_ids, spans, confs):
                answer = quid_to_q[q_id].answer

                if conf < 0.15:
                    text = " ".join(quid_to_q[q_id].get_context()[start:end+1])
                    if len(answer.answer_text) > 0:
                        score += max(f1_score(a, text) for a in answer.answer_text)
                    q_id_to_answers[q_id] = text
                else:
                    if len(answer.answer_text) == 0:
                        score += 1
                    q_id_to_answers[q_id] = ""

            print("Score: %.4f" % (score/len(q_ids)))

            with open(args.official_output + name + ".json", "w") as f:
                json.dump(q_id_to_answers, f)
#
if __name__ == "__main__":
    main()
    # tmp()




