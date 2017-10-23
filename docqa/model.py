from typing import List, Dict

from docqa.dataset import Dataset
from tensorflow import Tensor
from docqa.utils import ResourceLoader

from docqa.configurable import Configurable


class Prediction(object):
    """ Prediction from a model, subclasses should provide access to a tensor
     representations of the model's output """
    pass


class Model(Configurable):
    """
    Our most general specification of a model/neural network, for our purposes a model
    is basically a pair of functions
    1) a way to map a (unspecified) kind of python object to numpy tensors and
    2) a tensorflow function that maps those kinds of tensors to a set of (also unspecified) output tensors

    For convenience, models maintain a of set of input placeholders that clients can make use of to
    feed the tensorflow function (or reference to construct their own tensor inputs).

    Models have two stages of initialization. First it needs
    to be initialized with the training data using `init` (typically this does things like deciding what
    words/chars to train embeddings for). This should only be done once for this object's lifetime.

    Afterwards use `set_inputs` to specify the input format, this does things like determine the batch size
    or the vocabulary that will be used

    After initialiation, `encode` will produce map of placeholder -> numpy array
    which can be used directly as a feed dict for the output of `get_predictions`

    For more advanced usage, `get_predictions_for` can be used with any tensors of the
    same shape/dtype as the input place holders. Clients should pass in a dict mapping
    the placeholders to the input tensors they want to use instead.

    `get_predictions_for` methods behave like any other tensorflow function, in that it will
    load/initialize/reuse variables depending on the current tensorflow scope and can add
    to tf.collections. Our trainer method makes use of some of these collections, including:
        tf.GraphKeys.LOSSES
        tf.GraphKeys.REGULARIZATION_LOSSES
        tf.GraphKeys.SUMMARIES
        tf.GraphKeys.SAVEABLE_OBJECTS
        tf.GraphKeys.TRAINABLE_VARIABLES
        "monitor/*" collections, which will be summed, and the EMA result logged to tensorboard
    """

    @property
    def name(self):
        return self.__class__.__name__

    def init(self, train_data, resource_loader: ResourceLoader):
        raise NotImplementedError()

    def set_inputs(self, datasets: List[Dataset], resource_loader: ResourceLoader) -> List[Tensor]:
        raise NotImplementedError()

    def get_prediction(self) -> Prediction:
        return self.get_predictions_for({x: x for x in self.get_placeholders()})

    def get_placeholders(self) -> List[Tensor]:
        raise NotImplementedError()

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]) -> Prediction:
        raise NotImplementedError()

    def encode(self, examples, is_train: bool) -> Dict[Tensor, object]:
        raise NotImplementedError()
