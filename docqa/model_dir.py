import os
import pickle
from genericpath import exists
from os.path import isabs, join

import tensorflow as tf

from docqa.model import Model


class ModelDir(object):
    """ Wrapper for accessing a folder we are storing a model in"""

    def __init__(self, name: str):
        if isabs(name):
            print("WARNING!!!, using an absolute paths for models name can break restoring "
                  "the model in different directories after being checkpointed")
            # why is this even a thing?
        self.dir = name

    def get_model(self) -> Model:
        with open(join(self.dir, "model.pkl"), "rb") as f:
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
            if file.startswith("train_from_") and file.endswith("pkl"):
                step = int(file[11:file.rfind(".pkl")])
                if step > last_train_step:
                    last_train_step = step
                    last_train_file = join(self.dir, file)

        print("Resuming using the parameters stored in: " + last_train_file)
        with open(last_train_file, "rb") as f:
            return pickle.load(f)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.save_dir)

    def get_checkpoint(self, step):
        # I cant find much formal documentation on how to do this, but this seems to work
        return join(self.save_dir, "checkpoint-%d-%d" % (step, step))

    def get_best_weights(self):
        if exists(self.best_weight_dir):
            return tf.train.latest_checkpoint(self.best_weight_dir)
        return None

    def restore_checkpoint(self, sess, var_list=None, load_ema=True):
        """
        Restores either the best weights or the most recent checkpoint, assuming the correct
        variables have already been added to the tf default graph e.g., .get_prediction()
        has been called the model stored in `self`.
        Automatically detects if EMA weights exists, and if they do loads them instead
        """
        checkpoint = self.get_best_weights()
        if checkpoint is None:
            print("Loading most recent checkpoint")
            checkpoint = self.get_latest_checkpoint()
        else:
            print("Loading best weights")

        if load_ema:
            if var_list is None:
                # Same default used by `Saver`
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
                           tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)

            # Automatically check if there are EMA variables, if so use those
            reader = tf.train.NewCheckpointReader(checkpoint)
            ema = tf.train.ExponentialMovingAverage(0)
            ema_names = {ema.average_name(x): x for x in var_list
                         if reader.has_tensor(ema.average_name(x))}
            if len(ema_names) > 0:
                print("Found EMA weights, loading them")
                ema_vars = set(x for x in ema_names.values())
                var_list = {v.op.name: v for v in var_list if v not in ema_vars}
                var_list.update(ema_names)

        saver = tf.train.Saver(var_list)
        saver.restore(sess, checkpoint)


    @property
    def save_dir(self):
        # Stores training checkpoint
        return join(self.dir, "save")

    @property
    def best_weight_dir(self):
        # Stores training checkpoint
        return join(self.dir, "best-weights")

    @property
    def log_dir(self):
        return join(self.dir, "log")