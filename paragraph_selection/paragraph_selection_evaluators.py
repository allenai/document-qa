import numpy as np

from typing import List

from scipy.stats import rankdata

from evaluator import Evaluator, Evaluation
from model import ModelOutput


class AnyTopNEvaluator(Evaluator):
    def __init__(self, topn: List[int]):
        self.topn = sorted(topn)

    def tensors_needed(self, prediction: ModelOutput):
        return dict(paragraph_scores=prediction.prediction.paragraph_scores)

    def evaluate(self, data: List, true_len, **kwargs):
        paragraph_scores = kwargs["paragraph_scores"]
        sums = np.zeros(len(self.topn))

        for point_ix, point in enumerate(data):
            answer = point.answer
            top_picks = np.argsort(paragraph_scores[point_ix][:len(answer)])[::-1]
            for ix, n in enumerate(self.topn):
                if np.any(answer[top_picks[:n]] > 0):
                    sums[ix:] += 1
                    break

        sums /= true_len
        return Evaluation({("topn-any/top%d" % self.topn[i]): v for i,v in enumerate(sums)})


class PercentAnswerEvaluator(Evaluator):
    def __init__(self, topn: List[int]):
        self.topn = sorted(topn)

    def tensors_needed(self, prediction: ModelOutput):
        return dict(paragraph_scores=prediction.prediction.paragraph_scores)

    def evaluate(self, data: List, true_len, **kwargs):
        paragraph_scores = kwargs["paragraph_scores"]
        sums = np.zeros(len(self.topn))
        counts = np.zeros(len(self.topn))

        for point_ix, point in enumerate(data):
            answer = point.answer
            top_picks = np.argsort(paragraph_scores[point_ix][:len(answer)])[::-1]
            for ix, n in enumerate(self.topn):
                if n <= len(answer):
                    sums[ix] += answer[top_picks[n-1]] > 0
                    counts[ix] += 1

        sums /= counts
        return Evaluation({("topn-per/top%d" % self.topn[i]): v for i,v in enumerate(sums)})


class TotalAnswersEvaluator(Evaluator):
    def __init__(self, topn: List[int]):
        self.topn = sorted(topn)

    def tensors_needed(self, prediction: ModelOutput):
        return dict(paragraph_scores=prediction.prediction.paragraph_scores)

    def evaluate(self, data: List, true_len, **kwargs):
        paragraph_scores = kwargs["paragraph_scores"]
        sums = np.zeros(len(self.topn))

        for point_ix, point in enumerate(data):
            answer = point.answer
            top_picks = np.argsort(paragraph_scores[point_ix][:len(answer)])[::-1]
            for ix, n in enumerate(self.topn):
                sums[ix] += answer[top_picks[:n]].sum()

        sums /= true_len
        return Evaluation({("topn-total/%d" % self.topn[i]): v for i, v in enumerate(sums)})