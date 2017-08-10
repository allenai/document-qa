from typing import List, Iterable, Dict, Optional

import numpy as np
from configurable import Configurable
from tensorflow import Tensor

from dataset import Dataset
from utils import ResourceLoader


class Prediction(object):
    """ Prediction from a model, subclasses should provide access to a tensor
     representation of the model's output """
    pass


class ModelOutput(object):
    # TODO might be better to juse use hte "LOSS" graph collection
    def __init__(self, loss: Tensor, prediction: Prediction):
        self.loss = loss
        self.prediction = prediction


class Model(Configurable):
    """
    Our most general specification of a model/neural network, for our purposes a model
    is basically a pair of functions
    1) a way to map a (unspecified) kind of python object to numpy tensors (typically a batch of examples) and
    2) a tensorflow function that maps those kinds of tensors to (also unspecified) output tensors

    For convenience, models also maintain a of set of input placeholders that clients can make use of to
    feed the model (or reference to construct their own tensor inputs).

    Models have two stages of initialization. First it needs
    to be initialized with the training data using `init` (typically this does things like deciding what
    words/chars to train embeddings for). This should only be done once for this object's lifetime.

    Afterwards use `set_inputs` to specify the input format
    Once this is called, `encode` will produce map of placeholder -> numpy array
    which can be used directly as a feed dict or the output of
    `get_predictions` can be used with to get a prediction for the given placeholders.

    For more advanced usage, `get_predictions_for` can be used with any tensors of the
    same shape/dtype as the input place holders. Clients should pass in a dict mapping
    the placeholders to the input tensors they want to use instead.

    `get_predictions_for` methods behave like any other tensorflow function, in that it will
    load/initialize/reuse variables depending on the current tensorflow scope
    """
    # TODO not completely clear what the story will be w.r.t use of tf.collections

    @property
    def name(self):
        return self.__class__.__name__

    def init(self, train_data, resource_loader: ResourceLoader):
        raise NotImplementedError()

    def set_inputs(self, datasets: List[Dataset], resource_loader: ResourceLoader) -> List[Tensor]:
        raise NotImplementedError()

    def get_prediction(self) -> ModelOutput:
        return self.get_predictions_for({x: x for x in self.get_placeholders()})

    def get_placeholders(self) -> List[Tensor]:
        raise NotImplementedError()

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]) -> ModelOutput:
        raise NotImplementedError()

    def encode(self, examples, is_train: bool) -> Dict[Tensor, np.ndarray]:
        raise NotImplementedError()
