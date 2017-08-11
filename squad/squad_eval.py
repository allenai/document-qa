from typing import List, Union

import numpy as np

from data_processing.paragraph_qa import DocParagraphAndQuestion
from data_processing.qa_data import ParagraphAndQuestion
from data_processing.span_data import compute_span_f1, get_best_span, get_best_in_sentence_span
from evaluator import Evaluator, Evaluation
from squad.squad_official_evaluation import exact_match_score as squad_official_em_score
from squad.squad_official_evaluation import f1_score as squad_official_f1_score


"""
SQuAD specific evalation
"""


def squad_span_scores(data: List[Union[ParagraphAndQuestion, DocParagraphAndQuestion]], prediction):
    scores = np.zeros((len(data), 4))
    for i in range(len(data)):
        para = data[i]

        pred_span = tuple(prediction[i])
        pred_text = para.get_original_text(pred_span[0], pred_span[1])

        span_correct = False
        span_max_f1 = 0
        text_correct = 0
        text_max_f1 = 0
        answer = data[i].answer
        for (start, end), text in zip(answer.answer_spans, answer.answer_text):
            answer_span = (start, end)
            span_max_f1 = max(span_max_f1, compute_span_f1(answer_span, pred_span))
            if answer_span == pred_span:
                span_correct = True
            f1 = squad_official_f1_score(pred_text, text)
            correct = squad_official_em_score(pred_text, text)
            text_correct = max(text_correct, correct)
            text_max_f1 = max(text_max_f1, f1)

        scores[i] = [span_correct, span_max_f1, text_correct, text_max_f1]

    return scores


def squad_span_evaluation(data: List[ParagraphAndQuestion],
                           true_len: int, prediction, prefix=""):
    scores = squad_span_scores(data, prediction).sum(axis=0) / true_len
    return Evaluation({
        prefix + "accuracy": scores[0],
        prefix + "f1": scores[1],
        prefix + "text-accuracy": scores[2],
        prefix + "text-f1": scores[3]
    })


class SquadSpanEvaluator(Evaluator):
    def tensors_needed(self, model):
        return dict(p1=model.prediction.start_probs, p2=model.prediction.end_probs)

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, p1, p2):
        best_spans = [get_best_span(p1[i], p2[i]) for i in range(len(p1))]
        return squad_span_evaluation(data, true_len, [x[0] for x in best_spans], "span/")


class BoundedSquadSpanEvaluator(Evaluator):
    def __init__(self, bound: List[int], record_samples=False):
        self.bound = bound
        self.record_samples = record_samples

    def tensors_needed(self, model):
        return {str(b): model.prediction.get_best_span(b)[0] for b in self.bound}

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, **kwargs):
        ev = Evaluation({})
        for b in self.bound:
            best_spans = kwargs[str(b)]
            ev.add(squad_span_evaluation(data, true_len, best_spans, "b%d/"%b))
        return ev


class SentenceSpanEvaluator(Evaluator):

    def tensors_needed(self, model):
        return dict(p1=model.prediction.start_probs, p2=model.prediction.end_probs)

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, p1, p2):
        best_spans = [get_best_in_sentence_span(p1[i], p2[i], [len(x) for x in data[i].context]) for i in range(len(p1))]
        return squad_span_evaluation(data, true_len, [x[0] for x in best_spans], "sr/")


