import json
import os
import pickle
import shutil
import time
from datetime import datetime
from json import JSONEncoder
from os.path import exists, join, relpath, isabs
from threading import Thread
from typing import List, Union, Tuple, Optional, Dict

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from configurable import Configurable
from dataset import TrainingData, Dataset
from evaluator import Evaluator, run_evaluators, Evaluation, run_evaluators_with
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adam import AdamOptimizer

import configurable
from model import Model
from utils import NumpyEncoder, flatten_iterable


class ModelDir(object):
    def __init__(self, name: str):
        if isabs(name):
            print("WARNING!!!, using an absolute paths for models name will break restoring "
                  "the model if the model directory is used")
        self.dir = name

    def get_model(self) -> Model:
        with open(join(self.dir, "model.npy"), "rb") as f:
            return pickle.load(f)

    def get_eval_dir(self):
        answer_dir = join(self.dir, "answers")
        if not exists(answer_dir):
            os.mkdir(answer_dir)
        return answer_dir

    def get_last_train_params(self):
        last_train_file = None
        last_train_step = -1
        for file in os.listdir(self.dir):
            if file.startswith("train_from_") and file.endswith("npy"):
                step = int(file[11:file.rfind(".npy")])
                if step > last_train_step:
                    last_train_step = step
                    last_train_file = join(self.dir, file)

        print("Resuming using the parameters stored in: " + last_train_file)
        with open(last_train_file, "rb") as f:
            return pickle.load(f)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.save_dir)

    def get_checkpoint(self, step):
        return tf.train.checkpoint_exists(self.save_dir)

    @property
    def eval_dir(self):
        return join(self.dir, "eval")

    @property
    def save_dir(self):
        return join(self.dir, "save")

    @property
    def log_dir(self):
        return join(self.dir, "log")


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


# TODO might be nicer to just have a "Trainer" object
class TrainParams(Configurable):
    """ Parameters related to training """
    def __init__(self,
                 opt: SerializableOptimizer,
                 num_epochs: int,
                 eval_period: int,
                 log_period: int,
                 save_period: int,
                 eval_samples: Dict[str, Optional[int]]=None,
                 regularization_weight: Optional[float]=None,
                 train_aux_only: int = 0,
                 async_encoding: Optional[int]=None,
                 max_checkpoints_to_keep: int = 5,
                 loss_ema: Optional[float] = .999,
                 monitor_ema: float = .999,
                 ema: Optional[float]=None):
        self.async_encoding = async_encoding
        self.regularization_weight = regularization_weight
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.opt = opt
        self.ema = ema
        self.loss_ema = loss_ema
        self.monitor_ema = monitor_ema
        self.num_epochs = num_epochs
        self.eval_period = eval_period
        self.log_period = log_period
        self.save_period = save_period
        self.train_aux_only = train_aux_only
        self.eval_samples = eval_samples


def init(out: ModelDir, model: Model, override=False):
    import socket
    hostname = socket.gethostname()

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

    init = dict(date=datetime.now().strftime("%m%d-%H%M%S"),
                host=hostname)

    with open(join(out.dir, "init.json"), "w") as f:
        f.write(configurable.description_to_json(init, indent=2))
    with open(join(out.dir, "model.json"), "w") as f:
        f.write(configurable.description_to_json(model, indent=2))
    with open(join(out.dir, "model.npy"), "wb") as f:
        pickle.dump(model, f)


def save_train_start(out,
                     data: TrainingData,
                     global_step: int,
                     evaluators: List[Evaluator],
                     train_params: TrainParams,
                     notes: str):
    if notes is not None:
        with open(join(out, "train_from_%d_notes.text" % global_step), "w") as f:
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
        f.write(configurable.description_to_json(train, indent=2))
    with open(join(out, "train_from_%d.npy" % global_step), "wb") as f:
        pickle.dump(train, f)


def _build_train_ops(loss, train_params):
    """ Bulid ops we should run during training, including learning, EMA, and summary ops"""
    global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                  initializer=tf.constant_initializer(0), trainable=False)

    summary_tensor = tf.summary.merge([[tf.summary.tensor_summary("loss", loss)] +
                                       tf.get_collection(tf.GraphKeys.SUMMARIES)])
    train_objective = loss

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(regularizers) > 0:
        regularization_loss = tf.add_n(regularizers)
        if train_params.regularization_weight is not None:
            train_objective = train_objective + regularization_loss * train_params.regularization_weight
        else:
            train_objective = train_objective + regularization_loss
    else:
        regularization_loss = None

    auxillary_tasks = tf.get_collection("auxillary_losses")
    if len(auxillary_tasks) > 0:
        auxillary_loss = tf.add_n(auxillary_tasks)
        train_objective = train_objective + auxillary_loss
    else:
        auxillary_loss = None

    opt = train_params.opt.get()
    train_opt = opt.apply_gradients(opt.compute_gradients(train_objective), global_step=global_step)

    if train_params.ema is not None:
        ema = tf.train.ExponentialMovingAverage(decay=train_params.ema)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_opt]):
            # Run the old training op, then update the averages.
            train_opt = tf.group(ema_op)

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

    if train_params.loss_ema is not None:
        loss_ema = tf.train.ExponentialMovingAverage(decay=train_params.loss_ema, name="LossEMA", zero_debias=True)

        if regularization_loss is None and auxillary_loss is None:
            ema_op = loss_ema.apply([loss])
            train_opt = tf.group(train_opt, ema_op)
            ema_var = loss_ema.average(loss)
            summary_tensor = tf.summary.merge([tf.summary.scalar("training-ema/loss", ema_var), summary_tensor])
        else:
            to_track = [loss, train_objective] + [x for x in [regularization_loss, auxillary_loss] if
                                                  x is not None]
            ema_op = loss_ema.apply(to_track)
            train_opt = tf.group(train_opt, ema_op)
            tensor_vars = [
                tf.summary.scalar("training-ema/loss", loss_ema.average(loss)),
                tf.summary.scalar("training-ema/objective", loss_ema.average(train_objective)),
            ]
            if regularization_loss is not None:
                tensor_vars.append(tf.summary.scalar("training-ema/regularization-loss",
                                                     loss_ema.average(regularization_loss)))
            if auxillary_loss is not None:
                tensor_vars.append(tf.summary.scalar("training-ema/auxillary-loss",
                                                     loss_ema.average(auxillary_loss)))
            summary_tensor = tf.summary.merge([tensor_vars, summary_tensor])

    return loss, summary_tensor, train_opt, global_step


def continue_training(
          data: TrainingData,
          model: Model,
          train_params: TrainParams,
          evaluators: List[Evaluator],
          out: ModelDir,
          notes: str=None,
          dry_run=False):
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
          notes: str=None,
          dry_run=False):
    print("Initializing model at: " + out.dir)
    model.init(data.get_train_corpus(), data.get_resource_loader())

    if not dry_run:
        init(out, model, False)

    _train(model, data, None,
           True, train_params, evaluators, out, notes, dry_run)


def resume_training(out: ModelDir, notes: str=None, dry_run=False, start_eval=False):
    train_params = out.get_last_train_params()
    model = out.get_model()

    train_data = train_params["data"]

    # TODO make preprocess officially in the training_data API
    train_data.preprocess(4, 1000)

    evaluators = train_params["evaluators"]
    params = train_params["train_params"]

    latest = tf.train.latest_checkpoint(out.save_dir)
    if latest is None:
        raise ValueError()

    _train(model, train_data, latest, False, params, evaluators, out, notes, dry_run, start_eval)


def resume_training_with(
        data: TrainingData,
        out: ModelDir,
        train_params: TrainParams,
        evaluators: List[Evaluator],
        notes: str=None,
        dry_run: bool=False):

    with open(join(out.dir, "model.npy"), "rb") as f:
        model = pickle.load(f)
    latest = out.get_latest_checkpoint()

    _train(model, data, latest, False,
           train_params, evaluators, out, notes, dry_run)


def _train(model: Model,
           data: TrainingData,
           checkpoint: Union[str, None],
           save_start: bool,
           train_params: TrainParams,
           evaluators: List[Evaluator],
           out: ModelDir,
           notes=None,
           dry_run=False,
           start_eval=False):

    if train_params.async_encoding:
        _train_async(model, data, checkpoint, save_start, train_params,
                     evaluators, out, notes, dry_run, start_eval)
        return

    # spec the model for the current voc/input/batching
    train = data.get_train()
    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()

    print("Training on %d samples (%d) batches" % (train.get_n_examples(), len(train)))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, data.get_n_examples()) for name, data in eval_datasets.items()))

    model.set_inputs([train] + list(eval_datasets.values()), loader)

    print("Setting up model prediction / tf...")

    pred = model.get_prediction()
    loss, summary_tensor, train_opt, global_step = _build_train_ops(pred.loss, train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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

    # Make sure not bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
    tf.get_default_graph().finalize()

    print("Start training!")

    mean_loss = 0
    batch_time = 0
    on_step = sess.run(global_step)
    if save_start:
        save_train_start(out.dir, data, on_step, evaluators, train_params, notes)

    for epoch in range(train_params.num_epochs):
        for batch_ix, batch in enumerate(train.get_epoch()):
            t0 = time.perf_counter()
            on_step = sess.run(global_step) + 1  # +1 because all calculations are done after step

            get_summary = on_step % train_params.log_period == 0
            encoded = model.encode(batch, True)

            if get_summary:
                loss, summary, _ = sess.run([pred.loss, summary_tensor, train_opt], feed_dict=encoded)
            else:
                summary = None
                loss, _ = sess.run([pred.loss, train_opt], feed_dict=encoded)

            batch_time += time.perf_counter() - t0
            mean_loss += loss
            if get_summary:
                av_loss = mean_loss/train_params.log_period
                print("on epoch=%d batch=%d step=%d loss=%.5f time=%.3f" %
                      (epoch, batch_ix+1, on_step, av_loss, batch_time))
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]), on_step)
                summary_writer.add_summary(summary, on_step)
                # FIXME "averaged loss" should be removed in favour or EMA, still here since I used it for other models
                # ands its nice to be able to make comparisons
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="averaged-loss", simple_value=av_loss)]), on_step)
                mean_loss = 0
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
                    if n_samples is not None:
                        it, _ = data.get_samples(n_samples)
                    else:
                        it = data.get_epoch()
                    evaluation = run_evaluators_with(eval_tensors, model, sess, evaluators,
                                                     it, data.percent_filtered())

                    for s in evaluation.to_summaries(name + "-"):
                        summary_writer.add_summary(s, on_step)

                print("Evaluation took: %.3f seconds" % (time.perf_counter()-t0))

    saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)
    sess.close()


def test(model: Model, evaluators, datasets: List[Dataset], loader, checkpoint, ema=False, sample=None):
    print("Setting up model")

    model.set_inputs(datasets, loader)

    pred = model.get_prediction()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print("Restoring variables")
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    if ema:
        # FIXME This is a bit stupid, since we are loading variables twice, but I found it
        # a bit fiddly to load the variables directly....
        print("Restoring EMA variables")
        ema = tf.train.ExponentialMovingAverage(0)
        saver = tf.train.Saver({ema.average_name(x): x for x in tf.trainable_variables()})
        saver.restore(sess, checkpoint)

    # precompute the tensors we need to do evaluations so
    # they can be re-used between datasets
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    tf.get_default_graph().finalize()

    print("Begin evaluation")

    dataset_outputs = []
    for dataset in datasets:
        if sample is not None:
            total, it = dataset.get_batches(sample)
        else:
            total, it = dataset.get_epoch()
        batches = tqdm(it, total=total)
        dataset_outputs.append(run_evaluators_with(eval_tensors, model, sess, evaluators,
                                              batches, dataset.percent_filtered()))
    return dataset_outputs


def _train_async(model: Model,
           data: TrainingData,
           checkpoint: Union[str, None],
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

    print("Training on %d samples (%d) batches" % (train.get_n_examples(), len(train)))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, data.get_n_examples()) for name, data in eval_datasets.items()))

    # spec the model for the given datasets
    model.set_inputs([train] + list(eval_datasets.values()), loader)
    placeholders = model.get_placeholders()

    train_queue = tf.FIFOQueue(train_params.async_encoding, [x.dtype for x in placeholders], name="train_queue")
    eval_queue = tf.FIFOQueue(train_params.async_encoding, [x.dtype for x in placeholders], name="eval_queue")
    train_enqeue = train_queue.enqueue(placeholders)
    eval_enqueue = eval_queue.enqueue(placeholders)
    train_close = train_queue.close(True)
    eval_size = eval_queue.size()
    is_train = tf.placeholder(tf.bool, ())
    input_tensors = tf.cond(is_train, lambda: train_queue.dequeue(), lambda: eval_queue.dequeue())

    # tensorfow can't infer the shape for an unsized queue, so set it manually
    for input_tensor, pl in zip(input_tensors, placeholders):
        input_tensor.set_shape(pl.shape)
    pred = model.get_predictions_for(dict(zip(placeholders, input_tensors)))

    print("Setting up model prediction / tf...")

    loss, summary_tensor, train_opt, global_step = _build_train_ops(pred.loss, train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
    summary_writer = tf.summary.FileWriter(out.log_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

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

    if save_start:
        save_train_start(out.dir, data, sess.run(global_step), evaluators, train_params, notes)

    def enqueue_train():
        for epoch in range(train_params.num_epochs):
            for batch in train.get_epoch():
                feed_dict = model.encode(batch, True)
                sess.run(train_enqeue, feed_dict)

    train_enqueue_thread = Thread(target=enqueue_train)
    train_enqueue_thread.daemon = True  # Ensure we exit the program on an excpetion

    print("Start training!")

    batch_time = 0
    on_step = sess.run(global_step)

    train_dict = {is_train: True}
    eval_dict = {is_train: False}

    train_enqueue_thread.start()

    for epoch in range(train_params.num_epochs):
        for batch_ix in range(len(train)):
            t0 = time.perf_counter()
            on_step = sess.run(global_step) + 1  # +1 because all calculations are done after step

            get_summary = on_step % train_params.log_period == 0

            if get_summary:
                loss, summary, _ = sess.run([pred.loss, summary_tensor, train_opt], feed_dict=train_dict)
            else:
                summary = None
                loss, _ = sess.run([pred.loss, train_opt], feed_dict=train_dict)

            batch_time += time.perf_counter() - t0
            if summary is not None:
                print("on epoch=%d batch=%d step=%d, time=%.3f" %
                      (epoch, batch_ix + 1, on_step, batch_time))
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]), on_step)
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
                    evaluation = run_evaluators_async(eval_tensors, model, sess, evaluators,
                                                     data, n_samples, eval_dict, eval_enqueue,
                                                     name)
                    if sess.run(eval_size) != 0:
                        raise RuntimeError("Leftover elements in the eval queue!")
                    for s in evaluation.to_summaries(name + "-"):
                        summary_writer.add_summary(s, on_step)

                print("Evaluation took: %.3f seconds" % (time.perf_counter() - t0))
    sess.run(train_close)  # terminates the enqueue thread with an exception
    train_enqueue_thread.join()  # wait for it to die

    saver.save(sess, relpath(join(out.save_dir, "checkpoint-" + str(on_step))), global_step=global_step)
    sess.close()


def run_evaluators_async(tensors_needed: List[Dict], model: Model, sess: tf.Session,
                         evaluators: List[Evaluator], dataset, n_sample, feed_dict, enqueue_op,
                         name, tq=False) -> Evaluation:
    all_tensors_needed = list(set(flatten_iterable(x.values() for x in tensors_needed)))

    tensors = {x: [] for x in all_tensors_needed}

    data_used = []
    if n_sample is None:
        batches, n_batches = dataset.get_epoch(), len(dataset)
    else:
        batches, n_batches = dataset.get_samples(n_sample)

    def enqueue_eval():
        for data in batches:
            encoded = model.encode(data, False)
            data_used.append(data)
            sess.run(enqueue_op, encoded)

    th = Thread(target=enqueue_eval)
    th.daemon = True
    th.start()
    for _ in tqdm(range(n_batches), total=n_batches, desc=name, ncols=80):
        output = sess.run(all_tensors_needed, feed_dict=feed_dict)
        for i in range(len(all_tensors_needed)):
            tensors[all_tensors_needed[i]].append(output[i])
    th.join()

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
        true_len = len(data_used) * 1/(1 - dataset.percent_filtered())

    combined = None
    for ev, needed in zip(evaluators, tensors_needed):
        args = {k: tensors[v] for k, v in needed.items()}
        evaluation = ev.evaluate(data_used, true_len, **args)
        if combined is None:
            combined = evaluation
        else:
            combined.add(evaluation)

    return combined
