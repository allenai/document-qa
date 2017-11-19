from threading import Thread
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from docqa.configurable import Configurable
from docqa.data_processing.qa_training_data import ContextAndQuestion
from docqa.data_processing.span_data import compute_span_f1
from docqa.dataset import Dataset
from docqa.model import Model, Prediction
from docqa.squad.squad_official_evaluation import exact_match_score as squad_official_em_score
from docqa.squad.squad_official_evaluation import f1_score as squad_official_f1_score
from docqa.triviaqa.trivia_qa_eval import exact_match_score as triviaqa_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as triviaqa_f1_score
from docqa.utils import flatten_iterable


class Evaluation(object):
    """
    Evaluation of model, includes scalar summaries and per-example records
    """

    def __init__(self, scalars: Dict[str, Any], per_sample: Dict[str, List]=None):
        self.scalars = scalars
        self.per_sample = per_sample

    def add(self, other):
        for k in self.scalars:
            if k in other.scalars:
                raise ValueError("Two evaluations had the same scalar key: " + k)
        self.scalars.update(other.scalars)

        if self.per_sample is None:
            self.per_sample = other.per_sample
        elif other.per_sample is not None:
            for k in self.per_sample:
                if k in other.per_sample:
                    raise ValueError("Two evaluations had the same per sample key: " + k)
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
        processed by the model (e.g. too large) and were removed, but we still want to report
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


def squad_span_scores(data: List[ContextAndQuestion], prediction):
    scores = np.zeros((len(data), 4))
    for i in range(len(data)):
        para = data[i]

        pred_span = tuple(prediction[i])
        # For SQuAD, we expect to be working with data points that know how to
        # retrieve the untokenized "raw" text each span is associated with
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


def trivia_span_scores(data: List[ContextAndQuestion],
                       prediction):
    scores = np.zeros((len(data), 4))
    for i in range(len(data)):
        para = data[i]
        ans = para.answer

        pred_span = prediction[i]
        # For TriviaQA we have generally called join-on-spaces approach good enough, since the answers here
        # tend to be short and the gold standard has better normalization. Possibly could get a very
        # small gain using the original text
        pred_text = " ".join(para.get_context()[pred_span[0]:pred_span[1]+1])

        span_correct = False
        span_max_f1 = 0
        text_correct = 0
        text_max_f1 = 0

        for word_start, word_end in ans.answer_spans:
            answer_span = (word_start, word_end)
            span_max_f1 = max(span_max_f1, compute_span_f1(answer_span, pred_span))
            if answer_span == tuple(pred_span):
                span_correct = True

        for text in ans.answer_text:
            f1 = triviaqa_f1_score(pred_text, text)
            correct = triviaqa_em_score(pred_text, text)
            text_correct = max(text_correct, correct)
            text_max_f1 = max(text_max_f1, f1)

        scores[i] = [span_correct, span_max_f1, text_correct, text_max_f1]
    return scores


class SpanEvaluator(Evaluator):
    """
    Evaluate span based models, if text_eval is a set it should produce exactly
    the scores returned by the corresponding official evaluation scripts
    """

    def __init__(self, bound: List[int], text_eval: str=None):
        if text_eval is not None and text_eval not in ["squad", "triviaqa"]:
            raise ValueError()
        self.bound = bound
        self.text_eval = text_eval

    def tensors_needed(self, prediction):
        return {str(b): prediction.get_best_span(b)[0] for b in self.bound}

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kwargs):
        ev = Evaluation({})
        for b in self.bound:
            best_spans = kwargs[str(b)]
            if self.text_eval is None:
                scores = span_scores(data, best_spans)
            elif self.text_eval == "triviaqa":
                scores = trivia_span_scores(data, best_spans)
            elif self.text_eval == "squad":
                scores = squad_span_scores(data, best_spans)
            else:
                raise RuntimeError()

            scores = scores.sum(axis=0) / true_len

            prefix = "b%d/"%b
            out = {
                prefix + "accuracy": scores[0],
                prefix + "f1": scores[1],
            }
            if self.text_eval is not None:
                out[prefix + "text-em"] = scores[2]
                out[prefix + "text-f1"] = scores[3]

            ev.add(Evaluation(out))
        return ev


class MultiParagraphSpanEvaluator(Evaluator):
    """
    Measure error with multiple paragraphs per a question.

    Evaluation is a bit tricky in this case, since we are generally sampling paragraphs
    each epoch we can't report exact numbers as your would see when running the
    evaluation scripts. Instead we report some numbers aimed to get an approximate idea of what is going on:

    1: question-text-{em|f1}, accuracy on questions-document pairs (or just questions if `per_doc=False`)
       using all sampled paragraphs when taking the model's highest confidence answer.
       This tends to be an overly-confident estimate since the sampled paragraphs are usually biased
       towards using paragraphs that contain the correct answer
    2: paragraph-text-{em|f1}, accuracy on answer-containing paragraphs (if `paragraph_level=True`)
    3: The Kendel Tau relation between the model's confidence and the paragraph's f1/em score,
       (if `k_tau=True`) intended to measure how valid the model's confidence score is
       when it comes to ranking.
    """

    def __init__(self, bound: int, eval, paragraph_level=True, k_tau=True,
                 per_doc=True):
        if eval not in ["squad", "triviaqa"]:
            raise ValueError()
        self.bound = bound
        self.eval = eval
        self.paragraph_level = paragraph_level
        self.k_tau = k_tau
        self.per_doc = per_doc

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(span=span, score=score)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kwargs):
        best_spans = kwargs["span"]
        span_logits = kwargs["score"]
        if self.eval == "triviaqa":
            scores = trivia_span_scores(data, best_spans)
        elif self.eval == "squad":
            scores = squad_span_scores(data, best_spans)
        else:
            raise RuntimeError()

        has_answer = np.array([len(x.answer.answer_spans) > 0 for x in data])

        selected_paragraphs = {}
        for i, point in enumerate(data):
            if self.per_doc:
                key = (point.question_id, point.doc_id)
            else:
                key = point.question_id
            if key not in selected_paragraphs:
                selected_paragraphs[key] = i
            elif span_logits[i] > span_logits[selected_paragraphs[key]]:
                selected_paragraphs[key] = i
        selected_paragraphs = list(selected_paragraphs.values())

        out = {
            "question-text-em": scores[selected_paragraphs, 2].mean(),
            "question-text-f1": scores[selected_paragraphs, 3].mean(),
        }

        if self.k_tau:
            out["text-em-k-tau"] = kendalltau(span_logits, scores[:, 2])[0]
            out["text-f1-k-tau"] = kendalltau(span_logits, scores[:, 3])[0]

        if self.paragraph_level:
            out["paragraph-text-em"] = scores[has_answer, 2].mean()
            out["paragraph-text-f1"] = scores[has_answer, 3].mean()

        prefix = "b%d/" % self.bound
        return Evaluation({prefix+k: v for k,v in out.items()})

    def __setstate__(self, state):
        if "per_doc" not in state:
            state["per_doc"] = True
        super().__setstate__(state)


class ConfidenceSpanEvaluator(Evaluator):
    """
    Measure error + try to record some statistics on the model's confidence scores
    """

    def __init__(self, bound: List[int], rank_metric="k-tau", text_eval="triviaqa"):
        if text_eval not in ["squad", "triviaqa"]:
            raise ValueError()
        self.text_eval = text_eval
        self.bound = bound
        self.rank_metric = rank_metric

    def tensors_needed(self, prediction):
        spans, conf = prediction.get_best_span(self.bound)
        needed = dict(spans=spans, conf=conf)
        if hasattr(prediction, "none_prob"):
            needed["none_prob"] = prediction.none_prob
        return needed

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        if self.text_eval == "triviaqa":
            scores = trivia_span_scores(data, kargs["spans"])
        elif self.text_eval == "squad":
            scores = squad_span_scores(data, kargs["spans"])
        else:
            raise RuntimeError()

        has_answer = [len(x.answer.answer_spans) > 0 for x in data]
        aggregated_scores = scores[has_answer].mean(axis=0)
        prefix ="b%d/" % self.bound
        scalars = {
            prefix + "accuracy": aggregated_scores[0],
            prefix + "f1": aggregated_scores[1],
            prefix + "text-accuracy": aggregated_scores[2],
            prefix + "text-f1": aggregated_scores[3]
        }

        if self.rank_metric == "spr":
            metric = spearmanr
        elif self.rank_metric == "k-tau":
            metric = kendalltau
        else:
            raise ValueError()

        if "none_prob" in kargs:
            none_conf = kargs["none_prob"]
            scalars[prefix + "none-text-f1-" + self.rank_metric] = metric(none_conf, scores[:, 3])[0]
            scalars[prefix + "none-span-accuracy-" + self.rank_metric] = metric(none_conf, scores[:, 0])[0]

        conf = kargs["conf"]
        scalars[prefix + "score-text-f1-" + self.rank_metric] = metric(conf, scores[:, 3])[0]
        scalars[prefix + "score-span-accuracy-" + self.rank_metric] = metric(conf, scores[:, 0])[0]
        return Evaluation(scalars)


class EvaluatorRunner(object):
    """ Knows how to run a list of evaluators """

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
    """ Knows how to run a list of evaluators use a tf.Queue to feed in the data """

    def __init__(self, evaluators: List[Evaluator], model: Model, queue_size: int):
        placeholders = model.get_placeholders()
        self.eval_queue = tf.FIFOQueue(queue_size, [x.dtype for x in placeholders],
                                  name="eval_queue")
        self.enqueue_op = self.eval_queue.enqueue(placeholders)
        self.dequeue_op = self.eval_queue.dequeue()
        self.close_queue = self.eval_queue.close(True)

        # Queue in this form has not shape info, so we have to add it in back here
        for x, p in zip(placeholders, self.dequeue_op):
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

        def enqueue_eval():
            try:
                for data in batches:
                    encoded = self.model.encode(data, False)
                    data_used.append(data)
                    sess.run(self.enqueue_op, encoded)
            except Exception as e:
                sess.run(self.close_queue)  # Crash the main thread
                raise e
            # we should run out of batches and exit gracefully

        th = Thread(target=enqueue_eval)

        th.daemon = True
        th.start()
        for _ in tqdm(range(n_batches), total=n_batches, desc=name, ncols=80):
            output = sess.run(all_tensors_needed, feed_dict=feed_dict)
            for i in range(len(all_tensors_needed)):
                tensors[all_tensors_needed[i]].append(output[i])
        th.join()

        if sess.run(self.queue_size) != 0:
            raise RuntimeError("All batches should be been consumed")

        # flatten the input
        for k in all_tensors_needed:
            v = tensors[k]
            if len(k.shape) == 0:
                v = np.array(v)  # List of scalars -> array
            elif any(x is None for x in k.shape.as_list()[1:]):
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
