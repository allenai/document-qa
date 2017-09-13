import os
import pickle
from genericpath import exists
from os.path import isabs, join

import tensorflow as tf

from model import Model


class ModelDir(object):
    def __init__(self, name: str):
        if isabs(name):
            print("WARNING!!!, using an absolute paths for models name can break restoring "
                  "the model in different directories after being checkpointed")
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
        # I cant find much formal documentation on how to do this, but this seems to work
        return join(self.save_dir, "checkpoint-%d-%d" % (step, step))

    @property
    def eval_dir(self):
        return join(self.dir, "eval")

    @property
    def save_dir(self):
        return join(self.dir, "save")

    @property
    def log_dir(self):
        return join(self.dir, "log")