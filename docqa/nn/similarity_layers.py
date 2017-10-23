import tensorflow as tf

from docqa.configurable import Configurable
from docqa.nn.layers import get_keras_initialization, get_keras_activation


def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None or mem_mask is None:
        raise NotImplementedError()

    x_mask = tf.sequence_mask(x_mask, x_word_dim)
    mem_mask = tf.sequence_mask(mem_mask, key_word_dim)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask


class SimilarityFunction(Configurable):
    """
    Computes a pairwise score between elements in each sequence
    (batch, time1, dim1], (batch, time2, dim2) -> (batch, time1, time2)
    """
    def get_scores(self, tensor_1, tensor_2):
        raise NotImplementedError

    def get_one_sided_scores(self, tensor_1, tensor_2):
        return tf.squeeze(self.get_scores(tf.expand_dims(tensor_1, 1), tensor_2), squeeze_dims=[1])


class _WithBias(SimilarityFunction):
    def __init__(self, bias: bool):
        # Note since we typically do softmax on the result, having a bias is usually redundant
        self.bias = bias

    def get_scores(self, tensor_1, tensor_2):
        out = self._distance_logits(tensor_1, tensor_2)
        if self.bias:
            bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
            out += bias
        return out

    def _distance_logits(self, tensor_1, tensor_2):
        raise NotImplemented()


class DotProduct(_WithBias):
    """ Dot-Prod attention with scaling as seen in https://arxiv.org/pdf/1706.03762.pdf """

    def __init__(self, bias: bool=False, scale: bool=False):
        super().__init__(bias)
        self.scale = scale

    def _distance_logits(self, tensor_1, tensor_2):
        dots = tf.matmul(tensor_1, tensor_2, transpose_b=True)
        if self.scale:
            last_dim = dots.shape.as_list()[-1]
            if last_dim is None:
                last_dim = tf.cast(tf.shape(dots)[-1], tf.float32)
            dots /= tf.sqrt(last_dim)
        return dots


class DotProductProject(_WithBias):
    """ Dot-Prod attention while projecting the input layers """

    def __init__(self, project_size, bias: bool=False, scale: bool=False,
                 project_bias: bool=False, init="glorot_uniform", share_project=False):
        super().__init__(bias)
        self.project_bias = project_bias
        self.init = init
        self.scale = scale
        self.project_size = project_size
        self.share_project = share_project

    def _distance_logits(self, x1, x2):
        init = get_keras_initialization(self.init)

        project1 = tf.get_variable("project1", (x1.shape.as_list()[-1], self.project_size), initializer=init)
        x1 = tf.tensordot(x1, project1, [[2], [0]])

        if self.share_project:
            if x2.shape.as_list()[-1] != x1.shape.as_list()[-1]:
                raise ValueError()
            project2 = project1
        else:
            project2 = tf.get_variable("project2", (x2.shape.as_list()[-1], self.project_size), initializer=init)
        x2 = tf.tensordot(x2, project2, [[2], [0]])

        if self.project_bias:
            x1 += tf.get_variable("bias1", (1, 1, self.project_size), initializer=tf.zeros_initializer())
            x2 += tf.get_variable("bias2", (1, 1, self.project_size), initializer=tf.zeros_initializer())

        dots = tf.matmul(x1, x2, transpose_b=True)
        if self.scale:
            dots /= tf.sqrt(tf.cast(self.project_size, tf.float32))
        return dots


class BiLinearSum(_WithBias):

    def __init__(self, bias: bool=False, init="glorot_uniform"):
        self.init = init
        super().__init__(bias)

    def _distance_logits(self, x, keys):
        init = get_keras_initialization(self.init)
        key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

        x_w = tf.get_variable("x_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

        # Broadcasting will expand the arrays to (batch, x_len, key_len)
        return tf.expand_dims(x_logits, axis=2) + tf.expand_dims(key_logits, axis=1)


class BiLinear(_WithBias):

    def __init__(self, projected_size: int, activation="tanh", bias: bool=False,
                 init="glorot_uniform", shared_projection=False):
        self.init = init
        self.activation = activation
        self.shared_project = shared_projection
        self.projected_size = projected_size
        super().__init__(bias)

    def _distance_logits(self, x, keys):
        init = get_keras_initialization(self.init)
        key_w = tf.get_variable("key_w", shape=(keys.shape.as_list()[-1], self.projected_size), initializer=init, dtype=tf.float32)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len, projected_size)

        if self.shared_project:
            x_w = key_w
        else:
            x_w = tf.get_variable("x_w", shape=(x.shape.as_list()[-1], self.projected_size), initializer=init, dtype=tf.float32)

        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len, projected_size)

        summed = tf.expand_dims(x_logits, axis=2) + tf.expand_dims(key_logits, axis=1)  # (batch, key_len, x_len, poject_size)

        summed = get_keras_activation(self.activation)(summed)

        combine_w = tf.get_variable("combine_w", shape=self.projected_size, initializer=init, dtype=tf.float32)

        return tf.tensordot(summed, combine_w, axes=[[3], [0]])  # (batch, key_len, x_len)


class TriLinear(_WithBias):
    """ Function used by BiDaF, bi-linear with an extra component for the dots of the vectors """
    def __init__(self, init="glorot_uniform", bias=False):
        super().__init__(bias)
        self.init = init

    def _distance_logits(self, x, keys):
        init = get_keras_initialization(self.init)

        key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

        x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

        dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)

        # Compute x * dot_weights first, the batch mult with x
        x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
        dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

        return dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)

    @property
    def version(self):
        return 1
