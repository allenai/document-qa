import os
import pickle
import shutil
import time
from datetime import datetime
from os.path import exists, join, relpath
from threading import Thread
from typing import List, Union, Optional, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adam import AdamOptimizer

from docqa import configurable
from docqa.configurable import Configurable
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.dataset import TrainingData, Dataset
from docqa.evaluator import Evaluator, Evaluation, AysncEvaluatorRunner, EvaluatorRunner
from docqa.model import Model
from docqa.model_dir import ModelDir

"""
Contains the train-loop and test-loop for our models
"""


class SerializableOptimizer(Configurable):
    """ So we can record what tensorflow optimizer we used """

    def __init__(self, opt_name, params=None):
        self.params = params
        self.opt_name = opt_name

    def get_params(self):
        return dict(opt_name=self.opt_name, params=self.params)

    def get(self, name=None):
        params = {} if self.params is None else self.params
        if self.opt_name == "Adam":
            if name is None:
                return AdamOptimizer(**params)
            else:
                return AdamOptimizer(name=name, **params)
        elif self.opt_name == "Adadelta":
            if name is None:
                return AdadeltaOptimizer(**params)
            else:
                return AdadeltaOptimizer(name=name, **params)
        else:
            raise NotImplemented()


def init(out: ModelDir, model: Model, override=False):
    """ Save our intial setup into `out` """

    for dir in [out.save_dir, out.log_dir]:
        if os.path.exists(dir):
            if len(os.listdir(dir)) > 0:
                if override:
                    print("Clearing %d files/dirs that already existed in %s" % (len(os.listdir(dir)), dir))
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                else:
                    raise ValueError()
        else:
            os.makedirs(dir)

    # JSON config just so we always have a human-readable dump of what we are working with
    with open(join(out.dir, "model.json"), "w") as f:
        f.write(configurable.config_to_json(model, indent=2))

    # Actual model saved via pickle
    with open(join(out.dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


# TODO might be nicer to just have a "Trainer" object
class TrainParams(Configurable):
    """ Parameters related to training """

    def __init__(self,
                 opt: SerializableOptimizer,
                 num_epochs: int,
                 eval_period: int,
                 log_period: int,
                 save_period: int,
                 eval_samples: Dict[str, Optional[int]],
                 regularization_weight: Optional[float] = None,
                 async_encoding: Optional[int] = None,
                 max_checkpoints_to_keep: int = 5,
                 loss_ema: Optional[float] = .999,
                 eval_at_zero: bool = False,
                 monitor_ema: float = .999,
                 ema: Optional[float] = None,
                 best_weights: Optional[Tuple[str, str]] = None
                 ):
        """
        :param opt: Optimizer to use
        :param num_epochs: Number of epochs to train for
        :param eval_period: How many batches to train on between evaluations
        :param log_period: How many batches to train on between logging
        :param save_period: How many batches to train on between checkpointing
        :param eval_samples: How many samples to draw during evaluation, None of a full epoch
        :param regularization_weight: How highly to weight regulraization, defaults to 1
        :param async_encoding: Encoding batches in a seperate thread, and store in a queue of this size
        :param max_checkpoints_to_keep: Max number of checkpoints to keep during training
        :param loss_ema: EMA weights for monitoring the loss during training
        :param eval_at_zero: Run an evaluation cycle before any training
        :param monitor_ema: EMA weights for monitor functions
        :param ema: EMA to use on the trainable parameters
        :param best_weights: Store the weights with the highest scores on the given eval dataset/metric
        """
        self.async_encoding = async_encoding
        self.regularization_weight = regularization_weight
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.opt = opt
        self.eval_at_zero = eval_at_zero
        self.ema = ema
        self.loss_ema = loss_ema
        self.monitor_ema = monitor_ema
        self.num_epochs = num_epochs
        self.eval_period = eval_period
        self.log_period = log_period
        self.save_period = save_period
        self.eval_samples = eval_samples
        self.best_weights = best_weights


def save_train_start(out,
                     data: TrainingData,
                     global_step: int,
                     evaluators: List[Evaluator],
                     train_params: TrainParams,
                     notes: str):
    """ Record the training parameters we are about to use into `out`  """

    if notes is not None:
        with open(join(out, "train_from_%d_notes.txt" % global_step), "w") as f:
            f.write(notes)

    import socket
    hostname = socket.gethostname()
    train = dict(train_params=train_params,
                 data=data,
                 start_at=global_step,
                 evaluators=evaluators,
                 date=datetime.now().strftime("%m%d-%H%M%S"),
                 host=hostname)
    with open(join(out, "train_from_%d.json" % global_step), "w") as f:
        f.write(configurable.config_to_json(train, indent=2))
    with open(join(out, "train_from_%d.pkl" % global_step), "wb") as f:
        pickle.dump(train, f)


def _build_train_ops(train_params):
    """ Bulid ops we should run during training, including learning, EMA, and summary ops"""
    global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                  initializer=tf.constant_initializer(0), trainable=False)

    loss = tf.get_collection(tf.GraphKeys.LOSSES)
    if len(loss) == 0:
        raise RuntimeError("No losses found in losses collection")
    loss = tf.add_n(loss, name="loss")

    if len(tf.get_collection(tf.GraphKeys.SUMMARIES)) > 0:
        # Add any summaries client stored in SUMMARIES
        summary_tensor = tf.summary.merge([[tf.summary.tensor_summary("loss", loss)] +
                                           tf.get_collection(tf.GraphKeys.SUMMARIES)])
    else:
        summary_tensor = tf.summary.tensor_summary("loss", loss)

    train_objective = loss

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(regularizers) > 0:
        regularization_loss = tf.add_n(regularizers, name="regularization_loss")
        if train_params.regularization_weight is not None:
            train_objective = train_objective + regularization_loss * train_params.regularization_weight
        else:
            train_objective = train_objective + regularization_loss
    else:
        regularization_loss = None

    opt = train_params.opt.get()
    train_opt = opt.apply_gradients(opt.compute_gradients(train_objective), global_step=global_step)

    if train_params.ema is not None:
        ema = tf.train.ExponentialMovingAverage(decay=train_params.ema)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_opt]):
            # Run the old training op, then update the averages.
            train_opt = tf.group(ema_op)
    else:
        ema = None

    # Any collections starting with "monitor" are also added as summaries
    to_monitor = {}
    for col in tf.get_default_graph().get_all_collection_keys():
        if col.startswith("monitor"):
            v = tf.get_collection(col)
            if len(v) > 0:
                print("Monitoring: " + col)
                v = tf.add_n(v)
                to_monitor[col] = v

    if len(to_monitor) > 0:
        monitor_ema = tf.train.ExponentialMovingAverage(decay=train_params.monitor_ema, name="MonitorEMA",
                                                        zero_debias=True)
        train_opt = tf.group(train_opt, monitor_ema.apply(list(to_monitor.values())))
        summary_tensor = tf.summary.merge(
            [tf.summary.scalar(col, monitor_ema.average(v)) for col, v in to_monitor.items()] +
            [summary_tensor])

    # EMA for the loss and what we monitoring
    if train_params.loss_ema is not None:
        loss_ema = tf.train.ExponentialMovingAverage(decay=train_params.loss_ema, name="LossEMA", zero_debias=True)

        if regularization_loss is None:
            ema_op = loss_ema.apply([loss])
            train_opt = tf.group(train_opt, ema_op)
            ema_var = loss_ema.average(loss)
            summary_tensor = tf.summary.merge([tf.summary.scalar("training-ema/loss", ema_var), summary_tensor])
        else:
            to_track = [loss, train_objective, regularization_loss]
            ema_op = loss_ema.apply(to_track)
            train_opt = tf.group(train_opt, ema_op)
            tensor_vars = [
                tf.summary.scalar("training-ema/loss", loss_ema.average(loss)),
                tf.summary.scalar("training-ema/objective", loss_ema.average(train_objective)),
                tf.summary.scalar("training-ema/regularization-loss",
                                  loss_ema.average(regularization_loss))
                ]
            summary_tensor = tf.summary.merge([tensor_vars, summary_tensor])

    return loss, summary_tensor, train_opt, global_step, ema


def continue_training(
        data: TrainingData,
        model: Model,
        train_params: TrainParams,
        evaluators: List[Evaluator],
        out: ModelDir,
        notes: str = None,
        dry_run=False):
    """ Train an already existing model, or start for scatch """
    if not exists(out.dir) or os.listdir(out.dir) == 0:
        start_training(data, model, train_params, evaluators, out, notes, dry_run)
    else:
        print("Files already exist, loading most recent model")
        resume_training_with(data, out, train_params, evaluators, notes, dry_run)


def start_training(
        data: TrainingData,
        model: Model,
        train_params: TrainParams,
        evaluators: List[Evaluator],
        out: ModelDir,
        notes: str = None,
        initialize_from=None,
        dry_run=False):
    """ Train a model from scratch """
    if initialize_from is None:
        print("Initializing model at: " + out.dir)
        model.init(data.get_train_corpus(), data.get_resource_loader())
    # Else we assume the model has already completed its first phase of initialization

    if not dry_run:
        init(out, model, False)

    _train(model, data, None, initialize_from,
           True, train_params, evaluators, out, notes, dry_run)


def resume_training(out: ModelDir, notes: str = None, dry_run=False, start_eval=False):
    """ Resume training an existing model """

    train_params = out.get_last_train_params()
    model = out.get_model()

    train_data = train_params["data"]

    evaluators = train_params["evaluators"]
    params = train_params["train_params"]
    params.num_epochs = 24*3

    if isinstance(train_data, PreprocessedData):
        # TODO don't hard code # of processes
        train_data.preprocess(6, 1000)

    latest = tf.train.latest_checkpoint(out.save_dir)
    if latest is None:
        raise ValueError("No checkpoint to resume from found in " + out.save_dir)

    _train(model, train_data, latest, None, False, params, evaluators, out, notes, dry_run, start_eval)


def resume_training_with(
        data: TrainingData,
        out: ModelDir,
        train_params: TrainParams,
        evaluators: List[Evaluator],
        notes: str = None,
        dry_run: bool = False):
    """ Resume training an existing model with the specified parameters """
    with open(join(out.dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    latest = out.get_latest_checkpoint()
    if latest is None:
        raise ValueError("No checkpoint to resume from found in " + out.save_dir)

    _train(model, data, latest, None, False,
           train_params, evaluators, out, notes, dry_run)


def _train(model: Model,
           data: TrainingData,
           checkpoint: Union[str, None],
           parameter_checkpoint: Union[str, None],
           save_start: bool,
           train_params: TrainParams,
           evaluators: List[Evaluator],
           out: ModelDir,
           notes=None,
           dry_run=False,
           start_eval=False):
    if train_params.async_encoding:
        _train_async(model, data, checkpoint, parameter_checkpoint, save_start, train_params,
                     evaluators, out, notes, dry_run, start_eval)
        return

    if train_params.best_weights is not None:
        raise NotImplementedError

    # spec the model for the current voc/input/batching
    train = data.get_train()
    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()
    evaluator_runner = EvaluatorRunner(evaluators, model)

    print("Training on %d batches" % len(train))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, len(data)) for name, data in eval_datasets.items()))

    print("Init model...")
    model.set_inputs([train] + list(eval_datasets.values()), loader)

    print("Setting up model prediction / tf...")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    with sess.as_default():
        pred = model.get_prediction()
    evaluator_runner.set_input(pred)

    if parameter_checkpoint is not None:
        print("Restoring parameters from %s" % parameter_checkpoint)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, parameter_checkpoint)
        saver = None

    loss, summary_tensor, train_opt, global_step, _ = _build_train_ops(train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
    summary_writer = tf.summary.FileWriter(out.log_dir)

    # Load or initialize the model parameters
    if checkpoint is not None:
        print("Restoring training from checkpoint...")
        saver.restore(sess, checkpoint)
        print("Loaded checkpoint: " + str(sess.run(global_step)))
        return
    else:
        if parameter_checkpoint is not None:
            print("Initializing training variables...")
            vars = [x for x in tf.global_variables() if x not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            sess.run(tf.variables_initializer(vars))
        else:
            print("Initializing parameters...")
            sess.run(tf.global_variables_initializer())

    # Make sure no bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
    tf.get_default_graph().finalize()

    print("Start training!")

    on_step = sess.run(global_step)
    if save_start:
        summary_writer.add_graph(sess.graph, global_step=on_step)
        save_train_start(out.dir, data, on_step, evaluators, train_params, notes)

    if train_params.eval_at_zero:
        print("Running evaluation...")
        start_eval = False
        for name, data in eval_datasets.items():
            n_samples = train_params.eval_samples.get(name)
            evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples)
            for s in evaluation.to_summaries(name + "-"):
                summary_writer.add_summary(s, on_step)

    batch_time = 0
    for epoch in range(train_params.num_epochs):
        for batch_ix, batch in enumerate(train.get_epoch()):
            t0 = time.perf_counter()
            on_step = sess.run(global_step) + 1  # +1 because all calculations are done after step

            get_summary = on_step % train_params.log_period == 0
            encoded = model.encode(batch, True)

            if get_summary:
                summary, _, batch_loss = sess.run([summary_tensor, train_opt, loss], feed_dict=encoded)
            else:
                summary = None
                _, batch_loss = sess.run([train_opt, loss], feed_dict=encoded)

            if np.isnan(batch_loss):
                raise RuntimeError("NaN loss!")

            batch_time += time.perf_counter() - t0
            if get_summary:
                print("on epoch=%d batch=%d step=%d time=%.3f" %
                      (epoch, batch_ix + 1, on_step, batch_time))
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]),
                                           on_step)
                summary_writer.add_summary(summary, on_step)
                batch_time = 0

            # occasional saving
            if on_step % train_params.save_period == 0:
                print("Checkpointing")
                saver.save(sess, join(out.save_dir, "checkpoint-" + str(on_step)), global_step=global_step)

            # Occasional evaluation
            if (on_step % train_params.eval_period == 0) or start_eval:
                print("Running evaluation...")
                start_eval = False
                t0 = time.perf_counter()
                for name, data in eval_datasets.items():
                    n_samples = train_params.eval_samples.get(name)
                    evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples)
                    for s in evaluation.to_summaries(name + "-"):
                        summary_writer.add_summary(s, on_step)

                print("Evaluation took: %.3f seconds" % (time.perf_counter() - t0))

    saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)
    sess.close()


def _train_async(model: Model,
                 data: TrainingData,
                 checkpoint: Union[str, None],
                 parameter_checkpoint: Union[str, None],
                 save_start: bool,
                 train_params: TrainParams,
                 evaluators: List[Evaluator],
                 out: ModelDir,
                 notes=None,
                 dry_run=False,
                 start_eval=False):
    """ Train while encoding batches on a seperate thread and storing them in a tensorflow Queue, can
    be much faster then using the feed_dict approach """

    train = data.get_train()

    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()

    print("Training on %d batches" % len(train))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, len(data)) for name, data in eval_datasets.items()))

    # spec the model for the given datasets
    model.set_inputs([train] + list(eval_datasets.values()), loader)
    placeholders = model.get_placeholders()

    train_queue = tf.FIFOQueue(train_params.async_encoding, [x.dtype for x in placeholders], name="train_queue")
    evaluator_runner = AysncEvaluatorRunner(evaluators, model, train_params.async_encoding)
    train_enqeue = train_queue.enqueue(placeholders)
    train_close = train_queue.close(True)

    is_train = tf.placeholder(tf.bool, ())
    input_tensors = tf.cond(is_train, lambda: train_queue.dequeue(),
                            lambda: evaluator_runner.eval_queue.dequeue())

    # tensorfow can't infer the shape for an unsized queue, so set it manually
    for input_tensor, pl in zip(input_tensors, placeholders):
        input_tensor.set_shape(pl.shape)

    print("Init model...")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        pred = model.get_predictions_for(dict(zip(placeholders, input_tensors)))

    evaluator_runner.set_input(pred)

    if parameter_checkpoint is not None:
        print("Restoring parameters from %s" % parameter_checkpoint)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        saver = None

    print("Setting up model prediction / tf...")
    all_vars = tf.global_variables()

    loss, summary_tensor, train_opt, global_step, weight_ema = _build_train_ops(train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    if train_params.best_weights is not None:
        lst = all_vars
        if weight_ema is not None:
            for x in lst:
                v = weight_ema.average(x)
                if v is not None:
                    lst.append(v)
        best_weight_saver = tf.train.Saver(var_list=lst, max_to_keep=1)
        cur_best = None
    else:
        best_weight_saver = None
        cur_best = None

    saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
    summary_writer = tf.summary.FileWriter(out.log_dir)

    # Load or initialize the model parameters
    if checkpoint is not None:
        print("Restoring from checkpoint...")
        saver.restore(sess, checkpoint)
        print("Loaded checkpoint: " + str(sess.run(global_step)))
    else:
        print("Initializing parameters...")
        sess.run(tf.global_variables_initializer())

    # Make sure no bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
    tf.get_default_graph().finalize()

    if dry_run:
        return

    on_step = sess.run(global_step)

    if save_start:
        # summary_writer.add_graph(sess.graph, global_step=on_step)
        save_train_start(out.dir, data, sess.run(global_step), evaluators, train_params, notes)

    def enqueue_train():
        try:
            # feed data from the dataset iterator -> encoder -> queue
            for epoch in range(train_params.num_epochs):
                for batch in train.get_epoch():
                    feed_dict = model.encode(batch, True)
                    sess.run(train_enqeue, feed_dict)
        except tf.errors.CancelledError:
            # The queue_close operator has been called, exit gracefully
            return
        except Exception as e:
            # Crashes the main thread with a queue exception
            sess.run(train_close)
            raise e

    train_enqueue_thread = Thread(target=enqueue_train)
    train_enqueue_thread.daemon = True  # Ensure we exit the program on an excpetion

    print("Start training!")

    batch_time = 0

    train_dict = {is_train: True}
    eval_dict = {is_train: False}
    try:
        train_enqueue_thread.start()

        for epoch in range(train_params.num_epochs):
            for batch_ix in range(len(train)):
                t0 = time.perf_counter()
                on_step = sess.run(global_step) + 1
                get_summary = on_step % train_params.log_period == 0

                if get_summary:
                    summary, _, batch_loss = sess.run([summary_tensor, train_opt, loss], feed_dict=train_dict)
                else:
                    summary = None
                    _, batch_loss = sess.run([train_opt, loss], feed_dict=train_dict)

                if np.isnan(batch_loss):
                    raise RuntimeError("NaN loss!")

                batch_time += time.perf_counter() - t0
                if summary is not None:
                    print("on epoch=%d batch=%d step=%d, time=%.3f" %
                          (epoch, batch_ix + 1, on_step, batch_time))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]), on_step)
                    summary_writer.add_summary(summary, on_step)
                    batch_time = 0

                # occasional saving
                if on_step % train_params.save_period == 0:
                    print("Checkpointing")
                    saver.save(sess, join(out.save_dir, "checkpoint-" + str(on_step)), global_step=global_step)

                # Occasional evaluation
                if (on_step % train_params.eval_period == 0) or start_eval:
                    print("Running evaluation...")
                    start_eval = False
                    t0 = time.perf_counter()
                    for name, data in eval_datasets.items():
                        n_samples = train_params.eval_samples.get(name)
                        evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples, eval_dict)
                        for s in evaluation.to_summaries(name + "-"):
                            summary_writer.add_summary(s, on_step)

                        # Maybe save as the best weights
                        if train_params.best_weights is not None and name == train_params.best_weights[0]:
                            val = evaluation.scalars[train_params.best_weights[1]]
                            if cur_best is None or val > cur_best:
                                print("Save weights with current best weights (%s vs %.5f)" % (
                                    "None" if cur_best is None else ("%.5f" % cur_best), val))
                                best_weight_saver.save(sess, join(out.best_weight_dir, "best"), global_step=global_step)
                                cur_best = val

                    print("Evaluation took: %.3f seconds" % (time.perf_counter() - t0))
    finally:
        sess.run(train_close)  # terminates the enqueue thread with an exception

    train_enqueue_thread.join()

    saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)
    sess.close()


def test(model: Model, evaluators, datasets: Dict[str, Dataset], loader, checkpoint,
         ema=True, aysnc_encoding=None, sample=None) -> Dict[str, Evaluation]:
    print("Setting up model")
    model.set_inputs(list(datasets.values()), loader)

    if aysnc_encoding:
        evaluator_runner = AysncEvaluatorRunner(evaluators, model, aysnc_encoding)
        inputs = evaluator_runner.dequeue_op
    else:
        evaluator_runner = EvaluatorRunner(evaluators, model)
        inputs = model.get_placeholders()
    input_dict = {p: x for p, x in zip(model.get_placeholders(), inputs)}

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        pred = model.get_predictions_for(input_dict)
    evaluator_runner.set_input(pred)

    print("Restoring variables")
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    if ema:
        # FIXME This is a bit stupid, since we are loading variables twice, but I found it
        # a bit fiddly to load the variables directly....
        ema = tf.train.ExponentialMovingAverage(0)
        reader = tf.train.NewCheckpointReader(checkpoint)
        expected_ema_names = {ema.average_name(x): x for x in tf.trainable_variables()
                              if reader.has_tensor(ema.average_name(x))}
        if len(expected_ema_names) > 0:
            print("Restoring EMA variables")
            saver = tf.train.Saver(expected_ema_names)
            saver.restore(sess, checkpoint)

    tf.get_default_graph().finalize()

    print("Begin evaluation")

    dataset_outputs = {}
    for name, dataset in datasets.items():
        dataset_outputs[name] = evaluator_runner.run_evaluators(sess, dataset, name, sample, {})
    return dataset_outputs