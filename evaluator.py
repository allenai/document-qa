import json
from typing import List, Dict

import numpy as np
import tensorflow as tf

from configurable import Configurable
from data_processing.paragraph_qa import ParagraphAndQuestion
from model import ModelOutput, Model
from utils import NumpyEncoder, flatten_iterable


class Evaluation(object):

    def __init__(self, scalars, per_sample: Dict[str, List]=None):
        self.scalars = scalars
        self.per_sample = per_sample

    def add(self, other):
        for k in self.scalars:
            if k in other.scalars:
                raise ValueError()
        self.scalars.update(other.scalars)

        if self.per_sample is None:
            self.per_sample = other.per_sample
        elif other.per_sample is not None:
            for k in self.per_sample:
                if k in other.per_sample:
                    raise ValueError()
            self.per_sample.update(other.per_sample)

    def add_prefix(self, prefix):
        self.scalars = {prefix+k:v for k,v in self.scalars.items()}
        self.per_sample = {prefix+k:v for k,v in self.per_sample.items()}

    def to_summaries(self, prefix):
        return [tf.Summary(value=[tf.Summary.Value(tag=prefix + k, simple_value=v)]) for k,v in self.scalars.items()]


def log_evaluation(eval: Evaluation, data_used, output_file=None):
    for k,v in eval.scalars.items():
        print("%s: %.4f" % (k, v))

    if output_file is not None:
        if eval.per_sample is None:
            print("Output given, but no evalutor recorded per-samplee statistics")
        else:
            keys = list(eval.per_sample.keys())
            print("Saving %s for each question" % str(keys))
            values = np.array([eval.per_sample[k] for k in keys]).T
            with open(output_file, "w") as f:
                for i in range(len(data_used)):
                    point = data_used[i]
                    out = dict(question_id=point.question_id)
                    out.update((keys[j], values[i, j]) for j in range(len(keys)))
                    f.write(json.dumps(out, cls=NumpyEncoder))
                    f.write("\n")


class Evaluator(Configurable):
    """ Class to generate statistics on a model's output for some data"""

    def tensors_needed(self, prediction: ModelOutput):
        """ Return all tensor variables needed by this evaluator in a dict, the results will
        be passed into `build_summary` """
        pass

    def evaluate(self, input: List, true_len,  **kwargs) -> Evaluation:
        """ Build a summary given the input data `input` and the result of the variables requested
        from `tensors_needed`. `true_len` is the total number of examples seen (or an approximation)
        excludign any pre-filtering that was done, its used for the case where some examples could not be
        processed by the model (i.e. too large) and were removed, but we still want to report
        accurate percentages on the entire dataset. """
        pass


class LossEvaluator(Evaluator):

    def tensors_needed(self, prediction):
        return dict(loss=prediction.loss)

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, loss):
        return Evaluation({"loss": np.mean(loss)})


class RegularizerLossEvaluator(Evaluator):

    def tensors_needed(self, prediction):
        regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(regularizers) == 0:
            return {}
        else:
            return dict(reg=tf.add_n(regularizers))

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, reg=None):
        if reg is None:
            return Evaluation({})
        return Evaluation({"regularization-loss": np.mean(reg)})


class AuxillaryLossEvaluator(Evaluator):

    def tensors_needed(self, prediction):
        aux = tf.get_collection("auxillary_losses")
        if len(aux) == 0:
            return {}
        else:
            return dict(reg=tf.add_n(aux))

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, reg=None):
        if reg is None:
            return Evaluation({})
        return Evaluation({"auxillary_loss": np.mean(reg)})


def record_span_predictions(predictions, name):
    answers = []
    for i, ((start, end), val) in enumerate(predictions):
        answers.append(dict(start=int(start), end=int(end), val=float(val)))
    return Evaluation({}, {name: answers})


class RecordQuestionId(Evaluator):
    def tensors_needed(self, prediction: ModelOutput):
        return dict()

    def evaluate(self, data, true_len, **kwargs):
        return Evaluation({}, dict(question_id=[x.question_id for x in data]))


class RecordSpanPrediction(Evaluator):
    def __init__(self, bound: int, prefix=None):
        self.bound = bound
        self.prefix = prefix

    def tensors_needed(self, model):
        span, score = model.prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[ParagraphAndQuestion], true_len, **kargs):
        spans, model_scores = kargs["spans"], kargs["model_scores"]
        if self.prefix is None:
            prefix = "bound-%d-" % self.bound
        elif self.prefix == "":
            prefix = ""
        else:
            prefix = self.prefix + "-"
        span_key, score_key = ("%sspan-predictions" % prefix), ("%smodel-score" % prefix)
        results = {score_key: model_scores, span_key: spans}
        return Evaluation({}, results)


def run_evaluators(prediction: ModelOutput, model: Model, sess: tf.Session,
                   evaluators: List[Evaluator],  iter_batches, percent_filtered=None) -> Evaluation:
    tensors_needed = []
    for ev in evaluators:
        tensors_needed.append(ev.tensors_needed(prediction))
    return run_evaluators_with(tensors_needed, model, sess, evaluators, iter_batches, percent_filtered)


def run_evaluators_with(tensors_needed: List[Dict], model: Model, sess: tf.Session,
                   evaluators: List[Evaluator], iter_batches, percent_filtered=None) -> Evaluation:
    all_tensors_needed = list(set(flatten_iterable(x.values() for x in tensors_needed)))

    tensors = {x: [] for x in all_tensors_needed}

    data_used = []

    for batch in iter_batches:
        feed_dict = model.encode(batch, is_train=False)
        output = sess.run(all_tensors_needed, feed_dict=feed_dict)
        data_used += batch
        for i in range(len(all_tensors_needed)):
            tensors[all_tensors_needed[i]].append(output[i])

    # flatten the input
    for k in all_tensors_needed:
        v = tensors[k]
        if len(k.shape) == 0:
            v = np.array(v)  # List of scalars
        elif any(x is None for x in k.shape.as_list()):
            # Variable sized tensors, so convert to flat python-list
            v = flatten_iterable(v)
        else:
            v = np.concatenate(v, axis=0)  # concat along the batch dim
        tensors[k] = v

    if percent_filtered is None:
        true_len = len(data_used)
    else:
        true_len = len(data_used) * 1/(1 - percent_filtered)

    combined = None
    for ev, needed in zip(evaluators, tensors_needed):
        args = {k: tensors[v] for k, v in needed.items()}
        evaluation = ev.evaluate(data_used, true_len, **args)
        if combined is None:
            combined = evaluation
        else:
            combined.add(evaluation)

    return combined


# deprecated, here To not break picke
class SquadSpanEvaluator(Evaluator):
    def __init__(self):
        raise ValueError("Deprecated")


class TriviaQaBoundedSpanEvaluator(Evaluator):
    def __init__(self):
        raise ValueError("Deprecated")


class TfTriviaQaBoundedSpanEvaluator(Evaluator):
    def __init__(self):
        raise ValueError("Deprecated")
