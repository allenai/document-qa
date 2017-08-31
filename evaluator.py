import json
from threading import Thread, Event
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configurable import Configurable
from data_processing.multi_paragraph_qa import DocumentParagraph
from data_processing.qa_training_data import ContextAndQuestion
from data_processing.span_data import compute_span_f1
from dataset import Dataset
from model import Model, Prediction
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


class Evaluator(Configurable):
    """ Class to generate statistics on a model's output for some data"""

    def tensors_needed(self, prediction: Prediction):
        """ Return all tensor variables needed by this evaluator in a dict, the results will
        be passed into `build_summary` as numpy arrays """
        raise NotImplementedError()

    def evaluate(self, input: List, true_len,  **kwargs) -> Evaluation:
        """
        Build a summary given the input data `input` and the result of the variables requested
        from `tensors_needed`. `true_len` is the total number of examples seen (or an approximation)
        excluding any pre-filtering that was done, its used for the case where some examples could not be
        processed by the model (i.e. too large) and were removed, but we still want to report
        accurate percentages on the entire dataset.
        """
        raise NotImplementedError()


class LossEvaluator(Evaluator):

    def tensors_needed(self, _):
        return dict(loss=tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES)))

    def evaluate(self, data, true_len, loss):
        return Evaluation({"loss": np.mean(loss)})


class RegularizerLossEvaluator(Evaluator):

    def tensors_needed(self, _):
        regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(regularizers) == 0:
            return {}
        else:
            return dict(reg=tf.add_n(regularizers))

    def evaluate(self, data: List[ContextAndQuestion], true_len, reg=None):
        if reg is None:
            return Evaluation({})
        return Evaluation({"regularization-loss": np.mean(reg)})


class RecordQuestionId(Evaluator):
    def tensors_needed(self, _):
        return dict()

    def evaluate(self, data, true_len, **kwargs):
        out = dict(question_id=[x.question_id for x in data])
        if isinstance(data[0], DocumentParagraph):
            out.update(dict(doc_id=[x.doc_id for x in data],
                       start=np.array([x.start for x in data]),
                       end=np.array([x.end for x in data])))
        return Evaluation({}, out)


class SpanProbability(Evaluator):
    def __init__(self, sum=True):
        self.sum = sum

    def tensors_needed(self, prediction):
        return dict(p1=prediction.start_probs, p2=prediction.end_probs)

    def evaluate(self, data: List[ContextAndQuestion], true_len, p1, p2):
        start_probs = []
        end_probs = []
        for ix, point in enumerate(data):
            start_prob = 0
            end_prob = 0
            for start, end in point.answer.answer_spans:
                if self.sum:
                    start_prob += p1[ix][start]
                    end_prob += p2[ix][end]
                else:
                    start_prob = max(p1[ix][start], start_prob)
                    end_prob = max(p2[ix][end], end_prob)

            start_probs.append(start_prob)
            end_probs.append(end_prob)
        start_probs = np.array(start_probs)
        end_probs = np.array(end_probs)
        prefix = "span-prob/"
        return Evaluation({prefix + "start": np.mean(start_probs),
                           prefix + "span": np.mean(start_probs*end_probs),
                           prefix + "end": np.mean(end_probs)})


def span_scores(data: List[ContextAndQuestion], prediction):
    scores = np.zeros((len(data), 2))
    for i in range(len(data)):
        pred_span = tuple(prediction[i])

        span_correct = False
        span_max_f1 = 0
        answer = data[i].answer
        for (start, end) in answer.answer_spans:
            answer_span = (start, end)
            span_max_f1 = max(span_max_f1, compute_span_f1(answer_span, pred_span))
            if answer_span == pred_span:
                span_correct = True

        scores[i] = [span_correct, span_max_f1]

    return scores


class SpanEvaluator(Evaluator):
    def __init__(self, bound: List[int]):
        self.bound = bound

    def tensors_needed(self, prediction):
        return {str(b): prediction.get_best_span(b)[0] for b in self.bound}

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kwargs):
        ev = Evaluation({})
        for b in self.bound:
            best_spans = kwargs[str(b)]
            scores = span_scores(data, best_spans).sum(axis=0) / true_len
            prefix = "b%d/"%b
            bound_eval = Evaluation({
                prefix + "accuracy": scores[0],
                prefix + "f1": scores[1],
            })
            ev.add(bound_eval)
        return ev


class RecordSpanPrediction(Evaluator):
    def __init__(self, bound: int, prefix=None):
        self.bound = bound
        self.prefix = prefix

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
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


class EvaluatorRunner(object):
    def __init__(self, evaluators: List[Evaluator], model: Model):
        self.evaluators = evaluators
        self.tensors_needed = None
        self.model = model

    def set_input(self, prediction: Prediction):
        tensors_needed = []
        for ev in self.evaluators:
            tensors_needed.append(ev.tensors_needed(prediction))
        self.tensors_needed = tensors_needed

    def run_evaluators(self, sess: tf.Session, dataset: Dataset, name, n_sample=None, feed_dict=None) -> Evaluation:
        all_tensors_needed = list(set(flatten_iterable(x.values() for x in self.tensors_needed)))

        tensors = {x: [] for x in all_tensors_needed}

        if n_sample is None:
            batches, n_batches = dataset.get_epoch(), len(dataset)
        else:
            batches, n_batches = dataset.get_samples(n_sample)

        data_used = []

        for batch in tqdm(batches, total=n_batches, desc=name, ncols=80):
            feed_dict = self.model.encode(batch, is_train=False)
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

        percent_filtered = dataset.percent_filtered()
        if percent_filtered is None:
            true_len = len(data_used)
        else:
            true_len = len(data_used) * 1 / (1 - percent_filtered)

        combined = None
        for ev, needed in zip(self.evaluators, self.tensors_needed):
            args = {k: tensors[v] for k, v in needed.items()}
            evaluation = ev.evaluate(data_used, true_len, **args)
            if evaluation is None:
                raise ValueError(ev)
            if combined is None:
                combined = evaluation
            else:
                combined.add(evaluation)

        return combined


class AysncEvaluatorRunner(object):
    def __init__(self, evaluators: List[Evaluator], model: Model, queue_size: int):
        placeholders = model.get_placeholders()
        self.eval_queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders],
                                  name="eval_queue")
        self.enqueue_op = self.eval_queue.enqueue(placeholders)
        self.dequeue_op = self.eval_queue.dequeue()
        for x,p in zip(placeholders, self.dequeue_op):
            p.set_shape(x.shape)
        self.evaluators = evaluators
        self.queue_size = self.eval_queue.size()
        self.model = model
        self.tensors_needed = None

    def set_input(self, prediction: Prediction):
        tensors_needed = []
        for ev in self.evaluators:
            tensors_needed.append(ev.tensors_needed(prediction))
        self.tensors_needed = tensors_needed

    def run_evaluators(self, sess: tf.Session, dataset, name, n_sample, feed_dict) -> Evaluation:
        all_tensors_needed = list(set(flatten_iterable(x.values() for x in self.tensors_needed)))

        tensors = {x: [] for x in all_tensors_needed}

        data_used = []
        if n_sample is None:
            batches, n_batches = dataset.get_epoch(), len(dataset)
        else:
            batches, n_batches = dataset.get_samples(n_sample)

        enqueue_error = Event()

        def enqueue_eval():
            try:
                for data in batches:
                    encoded = self.model.encode(data, False)
                    data_used.append(data)
                    sess.run(self.enqueue_op, encoded)
            except Exception as e:
                enqueue_error.set()
                raise e

        th = Thread(target=enqueue_eval)

        th.daemon = True
        th.start()
        for _ in tqdm(range(n_batches), total=n_batches, desc=name, ncols=80):
            if enqueue_error.is_set():
                raise ValueError("Enqueue thread crashed")
            output = sess.run(all_tensors_needed, feed_dict=feed_dict)
            for i in range(len(all_tensors_needed)):
                tensors[all_tensors_needed[i]].append(output[i])
        th.join()

        if sess.run(self.queue_size) != 0:
            raise RuntimeError()

        # flatten the input
        for k in all_tensors_needed:
            v = tensors[k]
            if len(k.shape) == 0:
                v = np.array(v)  # List of scalars -> array
            elif any(x is None for x in k.shape.as_list()):
                # Variable sized tensors, so convert to flat python-list
                v = flatten_iterable(v)
            else:
                v = np.concatenate(v, axis=0)  # concat along the batch dim
            tensors[k] = v

        # flatten the data if it consists of batches
        if isinstance(data_used[0], List):
            data_used = flatten_iterable(data_used)

        if dataset.percent_filtered() is None:
            true_len = len(data_used)
        else:
            true_len = len(data_used) * 1 / (1 - dataset.percent_filtered())

        combined = None
        for ev, needed in zip(self.evaluators, self.tensors_needed):
            args = {k: tensors[v] for k, v in needed.items()}
            evaluation = ev.evaluate(data_used, true_len, **args)
            if combined is None:
                combined = evaluation
            else:
                combined.add(evaluation)

        return combined
