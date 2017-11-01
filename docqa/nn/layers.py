from typing import Optional, Union, List, Callable

import numpy as np
import tensorflow as tf
from docqa.configurable import Configurable
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import initializers
from tensorflow.python.layers.core import fully_connected

from docqa.model import Prediction
from docqa.nn.ops import dropout, exp_mask


def _wrap_init(init_fn):
    def wrapped(shape, dtype=None, partition_info=None):
        if partition_info is not None:
            raise ValueError()
        return init_fn(shape, dtype)
    return wrapped


def get_keras_initialization(name: Union[str, Callable]):
    if name is None:
        return None
    return _wrap_init(initializers.get(name))


def get_keras_activation(name: str):
    return activations.get(name)


"""
Basic layers, for our purposes layers act as serializable tensorflow functions, 
they are allowed to create variables, add to collections, ect.  
Layers should be immutable and behave like pure functions
that modify the tensorflow graph. Layers should also implement Configurable and be serialized  by pickle.

The top levels Layers define different tensor->tensor transforms, they exist so models can use
type hints to specify what components they need.   

(Most) layers are expected to handle `is_train` and `mask` parameters to deal with masked input
and to allow subclasses to implement test/train specific behaviour (usually dropout). Masks
are always assumed to be in the integer/sequence length format.
"""


class SequenceMapper(Configurable):
    """ (batch, time, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class Mapper(SequenceMapper):
    """ (dim1, dim2, ...., input_dim) -> (im1, dim2, ...., output_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class Updater(Mapper):
    """ (dim1, dim2, ...., input_dim) -> (im1, dim2, ...., input_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class MergeLayer(Configurable):
    """
    (dim1, dim2, ...., dimN, input_dim1),  (dim1, dim2, ...., dimN, input_dim2) ->
        (dim1, dim2, ...., dimN, output_dim)
    """
    def apply(self, is_train, tensor1, tensor2):
        raise NotImplemented()


class FixedMergeLayer(Configurable):
    """ (batch, time, in_dim) (batch, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, tensor, fixed_tensor, mask) -> tf.Tensor:
        raise NotImplemented()


class SequenceMapperWithContext(Configurable):
    """ (batch, time, in_dim) (batch, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, x, c, mask=None):
        raise NotImplementedError()


class Activation(Updater):
    """ This layer should not add to collection or create variables """

    def apply(self, is_train, x, mask=None):
        return self(x)

    def __call__(self, x):
        raise NotImplementedError()


class AttentionMapper(Configurable):
    """ (batch, time1, dim1), (batch, time1, dim2) (batch, time2, dim3) -> (batch, time1, out_dim) """
    def apply(self, is_train, x, keys, memories, mask=None, memory_mask=None):
        raise NotImplementedError()


class SequenceBiMapper(Configurable):
    """ (batch, time, in_dim) -> (batch, time, out_dim1), (batch, time, out_dim2) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class SequenceEncoder(Configurable):
    """ (batch, time, in_dim) -> (batch, out_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class Encoder(SequenceEncoder):
    """
    reduce the second to last dimension
     (dim1, dim2, ..., dimN, in_dim) -> (dim1, dim2, ..., dim(N-1), out_dim)
     mask should be an sequence length mask of dim one less then `x`
     """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class SequenceMultiEncoder(Configurable):
    """ (batch, time, in_dim) -> (batch, n_encodings, out_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class SqueezeLayer(Configurable):
    """
     removes the last dimension
    (dim1, dim2, dim3, input_dim) -> (dim1, dim2, dim3)
     """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class SequencePredictionLayer(Configurable):
    """ (batch, time, vec), answer -> Prediction """
    def apply(self, is_train, x, answer: List, mask=None) -> Prediction:
        return NotImplemented()


class AttentionPredictionLayer(Configurable):
    """ (batch, time1, vec1), (batch, time2, vec2), answers  -> Prediction """
    def apply(self, is_train, keys, context, answer: List, mask=None, memory_mask=None) -> Prediction:
        raise NotImplementedError()


class ResidualLayer(Mapper):

    def __init__(self, other: Union[Mapper, SequenceMapper]):
        self.other = other

    def apply(self, is_train, x, mask=None):
        return x + self.other.apply(is_train, x, mask)


class ProjectedResidualLayer(Mapper):

    def __init__(self, other: Union[Mapper, SequenceMapper]):
        self.other = other

    def apply(self, is_train, x, mask=None):
        out = self.other.apply(is_train, x, mask)
        w = tf.get_variable("project_w", (x.shape.as_list()[-1], out.shape.as_list()[-1]))
        return out + tf.tensordot(x, w, axes=[[len(x.shape)-1], [0]])


class MergeWith(SequenceMapper):

    def __init__(self, mapper: SequenceMapper, merge: MergeLayer=None):
        self.mapper = mapper
        self.merge = merge

    def apply(self, is_train, x, mask=None):
        with tf.variable_scope("map"):
            mapped = self.mapper.apply(is_train, x, mask)
        if self.merge is None:
            return tf.concat([x, mapped], axis=2)
        else:
            with tf.variable_scope("merge"):
                return self.merge.apply(is_train, x, mapped)


class LeakyRelu(Activation):
    def __init__(self, reduce_factor: float=0.3):
        self.reduce_factor = reduce_factor

    def __call__(self, x):
        return tf.where(x > 0, x, x * self.reduce_factor)


class ParametricRelu(Updater):
    def __init__(self, init=0):
        self.init = init

    def apply(self, is_train, x, mask=None):
        w = tf.get_variable("prelu", x.shape.as_list()[-1],
                            initializer=tf.constant_initializer(self.init))
        for i in range(len(x.shape)-1):
            w = tf.expand_dims(w, 0)
        return tf.where(x > 0, x, x * w)


class ConcatLayer(MergeLayer):
    def apply(self, is_train, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        return tf.concat([tensor1, tensor2], axis=len(tensor1.shape)-1)


class ConcatWithProduct(MergeLayer):
    def apply(self, is_train, tensor1, tensor2) -> tf.Tensor:
        return tf.concat([tensor1, tensor2, tensor1 * tensor2], axis=len(tensor1.shape) - 1)


class DotMerge(MergeLayer):
    def apply(self, is_train, tensor1, tensor2) -> tf.Tensor:
        return tensor1 * tensor2


class ConcatWithProductProj(MergeLayer):
    def __init__(self, n_project, init="glorot_uniform", dots=True, scale=True):
        self.n_project = n_project
        self.init = init
        self.scale = scale
        self.dots = dots

    def apply(self, is_train, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        init = get_keras_initialization(self.init)
        w1 = tf.get_variable("w1", (tensor1.shape.as_list()[-1], self.n_project), initializer=init)
        project1 = tf.tensordot(tensor1, w1, [[len(tensor1.shape)-1], [0]])
        if self.scale:
            project1 /= np.sqrt(self.n_project)

        w2 = tf.get_variable("w2", (tensor2.shape.as_list()[-1], self.n_project), initializer=init)
        project2 = tf.tensordot(tensor2, w2, [[len(tensor1.shape)-1], [0]])
        if self.scale:
            project2 /= np.sqrt(self.n_project)

        elements = [tensor1, tensor2, project1 * project2]
        if self.dots:
            elements.append(tensor1 * tensor2)

        return tf.concat(elements, axis=len(tensor1.shape) - 1)


class ConcatOneSidedProduct(MergeLayer):
    def __init__(self, init="glorot_uniform", scale=True, include_unscaled=True):
        self.init = init
        self.include_unscaled = include_unscaled
        self.scale = scale

    def apply(self, is_train, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        init = get_keras_initialization(self.init)
        w1 = tf.get_variable("w1", (tensor1.shape.as_list()[-1], tensor2.shape.as_list()[-1]), initializer=init)
        project1 = tf.tensordot(tensor1, w1, [[len(tensor1.shape)-1], [0]])
        if self.scale:
            project1 /= np.sqrt(tensor1.shape.as_list()[-1])
        project1 *= tensor2

        elements = [tensor1, project1]
        if self.include_unscaled:
            elements.append(tensor2)

        return tf.concat(elements, axis=len(tensor1.shape) - 1)


class FullyConnected(Mapper):
    def __init__(self, n_out,
                 w_init="glorot_uniform",
                 activation: Union[str, Updater, None]="relu",
                 bias=True):
        self.w_init = w_init
        self.activation = activation
        self.n_out = n_out
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        bias = (self.bias is None) or self.bias  # for backwards compat
        if isinstance(self.activation, Updater):
            out = fully_connected(x, self.n_out,
                                   use_bias=bias,
                                   activation=None,
                                   kernel_initializer=get_keras_initialization(self.w_init))
            with tf.variable_scope("activation"):
                return self.activation.apply(is_train, out, mask)
        else:
            return fully_connected(x, self.n_out,
                                   use_bias=bias,
                                   activation=get_keras_activation(self.activation),
                                   kernel_initializer=get_keras_initialization(self.w_init))


class FullyConnectedDotProject(Mapper):
    def __init__(self, n_out, n_project, w_init="glorot_uniform",
                 activation="relu", bias=True):
        self.w_init = w_init
        self.n_project = n_project
        self.activation = activation
        self.n_out = n_out
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        bias = (self.bias is None) or self.bias  # for backwards compat
        return fully_connected(x, self.n_out,
                               use_bias=bias,
                               activation=get_keras_activation(self.activation),
                               kernel_initializer=_wrap_init(initializers.get(self.w_init)))


class FullyConnectedUpdate(Updater):
    def __init__(self,  w_init="glorot_uniform",
                 residual=False,
                 activation="relu", bias=True):
        self.w_init = w_init
        self.activation = activation
        self.bias = bias
        self.residual = residual

    def apply(self, is_train, x, mask=None):
        bias = (self.bias is None) or self.bias
        out = fully_connected(x, x.shape.as_list()[-1],
                               use_bias=bias,
                               activation=get_keras_activation(self.activation),
                               kernel_initializer=get_keras_initialization(self.w_init))
        if self.residual:
            out += x
        return out


class ActivationLayer(Updater):
    def __init__(self, activation="relu", bias=True):
        self.activation = activation
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        if self.bias:
            x += tf.get_variable("bias", (1, 1, x.shape.as_list()[-1]), initializer=tf.zeros_initializer())
        return get_keras_activation(self.activation)(x)


class ProjectLayer(Updater):
    def __init__(self, w_init="glorot_uniform", activation="relu", bias=True):
        self.w_init = w_init
        self.activation = activation
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        return fully_connected(x, x.shape.as_list()[-1],
                               use_bias=self.bias,
                               activation=activations.get(self.activation),
                               kernel_initializer=_wrap_init(initializers.get(self.w_init)))


class GatingLayer(Updater):
    def __init__(self, init="glorot_uniform", bias: Optional[float]=1.0):
        self.init = init
        self.bias = bias

    def apply(self, is_train, x, mask=None):
        gated = fully_connected(x, x.shape.as_list()[-1], activation=tf.nn.sigmoid,
                                bias_initializer=tf.constant_initializer(self.bias) if self.bias else None,
                                kernel_initializer=get_keras_initialization(self.init), name="compute-gate")
        return gated * x


class HighwayLayer(Updater):
    def __init__(self, init="glorot_uniform", activation="tanh"):
        self.init = init
        self.activation = activation

    def apply(self, is_train, x, mask=None):
        n_out = x.shape.as_list()[-1]
        init = get_keras_initialization(self.init)
        activation = get_keras_activation(self.activation)
        trans = fully_connected(x, n_out, activation=activation, kernel_initializer=init, name="highway")
        gate = fully_connected(x, n_out, activation=tf.nn.sigmoid, kernel_initializer=init, name="gate")
        return gate * trans + (1 - gate) * x


class MapperSeq(Mapper):
    def __init__(self, *layers: Mapper):
        self.layers = layers

    def apply(self, is_train, x, mask=None):
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("layer_" + str(i)):
                x = layer.apply(is_train, x, mask)
        return x

    def get_params(self):
        return dict(layers=[x.get_params() for x in self.layers])


class SequenceMapperSeq(SequenceMapper):
    def __init__(self, *layers: SequenceMapper):
        self.layers = layers

    def apply(self, is_train, x, mask=None):
        for i, layer in enumerate(self.layers):
            with tf.variable_scope("layer_" + str(i)):
                x = layer.apply(is_train, x, mask)
        return x

    def get_params(self):
        return dict(layers=[x.get_config() for x in self.layers])


class NullMapper(Updater):
    def apply(self, is_train, x, mask=None):
        return x


class NullBiMapper(SequenceBiMapper):
    def apply(self, is_train, x, mask=None):
        return x, x


class FirstDimWeightedSum(Mapper):
    def apply(self, is_train, x, mask=None):
        s = x.shape.as_list()[1]
        w = tf.get_variable("w", (s,), dtype=tf.float32,
                            initializer=tf.constant_initializer(s / 3.0))
        b = tf.get_variable("b", (), dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return tf.tensordot(x, w, [[1], [0]]) + b


class FirstDimAverage(Mapper):
    def apply(self, is_train, x, mask=None):
        return tf.reduce_mean(x, axis=1)


class FirstDimProject(Mapper):
    def __init__(self, n_out):
        self.n_out = n_out

    def apply(self, is_train, x, mask=None):
        _, d1, _, d2 = x.shape.as_list()
        w = tf.get_variable("w", (d1, d2, self.n_out), dtype=tf.float32)
        return tf.tensordot(x, w, [[1, 3], [0, 1]])


class IndependentBiMapper(SequenceBiMapper):
    def __init__(self, first_layer: SequenceMapper, second_layer: SequenceMapper):
        self.first_layer = first_layer
        self.second_layer = second_layer

    def apply(self, is_train, x, mask=None):
        with tf.variable_scope("part1"):
            m1 = self.first_layer.apply(is_train, x, mask)
        with tf.variable_scope("part2"):
            m2 = self.second_layer.apply(is_train, x, mask)
        return m1, m2


class ChainBiMapper(SequenceBiMapper):
    def __init__(self, first_layer: SequenceMapper, second_layer: SequenceMapper):
        self.first_layer = first_layer
        self.second_layer = second_layer

    def apply(self, is_train, x, mask=None):
        with tf.variable_scope("out"):
            m1 = self.first_layer.apply(is_train, x, mask)
        with tf.variable_scope("chained-out"):
            m2 = self.second_layer.apply(is_train, tf.concat([x, m1], axis=2), mask)
        return m1, m2


class MapMulti(Configurable):
    """ Applies a layer to multiple inputs, possibly sharing parameters """

    def __init__(self, layer: SequenceMapper, share: bool=True):
        self.layer = layer
        self.share = share

    def apply(self, is_train, *inputs):
        if self.share:
            with tf.variable_scope("map"):
                output = [self.layer.apply(is_train, inputs[0][0], inputs[0][1])]
            with tf.variable_scope("map", reuse=True):
                for i in range(1, len(inputs)):
                    output.append(self.layer.apply(is_train, inputs[i][0], inputs[1][1]))
        else:
            output = []
            for i,(x,mask) in enumerate(inputs):
                with tf.variable_scope("map%d_%s" % (i, x.name)):
                    output.append(self.layer.apply(is_train, x, mask))
        return output


class MapMemoriesBiMapper(SequenceBiMapper):
    def __init__(self, map: SequenceMapper):
        self.map = map

    def apply(self, is_train, x, mask=None):
        with tf.variable_scope("build-memories"):
            mem = self.map.apply(is_train, x)
        return x, mem


class TileAndMerge(SequenceMapperWithContext):
    def __init__(self, merge: MergeLayer):
        self.merge = merge

    def apply(self, is_train, x, c, mask=None, context_mask=None):
        tiled_c = tf.tile(tf.expand_dims(c, 1), [1, tf.shape(x)[1], 1])
        with tf.variable_scope("merge"):
            return self.merge.apply(is_train, x, tiled_c)


class FullyConnectedContextMerge(SequenceMapperWithContext):
    def __init__(self, output_size, init="glorot_uniform", activation="tanh",
                 use_dots=False, keep_probs=1, context_keep_probs=1):
        self.output_size = output_size
        self.activation = activation
        self.init = init
        self.context_keep_probs = context_keep_probs
        self.keep_probs = keep_probs
        self.use_dots = use_dots

    def apply(self, is_train, x, c, mask=None, context_mask=None):
        x = dropout(x, self.keep_probs, is_train)
        c = dropout(c, self.context_keep_probs, is_train)
        init = get_keras_initialization(self.init)
        x_w = tf.get_variable("merge_x_weights", (x.shape.as_list()[-1], self.output_size), initializer=init)
        c_w = tf.get_variable("merge_context_weights", (c.shape.as_list()[-1], self.output_size), initializer=init)
        output = tf.tensordot(x, x_w, axes=[[2], [0]]) + tf.expand_dims(tf.matmul(c, c_w), 1)
        if self.use_dots:
            dots = tf.einsum("aij,aj->aij", x, c)
            dot_w = tf.get_variable("dot_weights", (c.shape.as_list()[-1], self.output_size), initializer=init)
            output += tf.tensordot(dots, dot_w, axes=[[2], [0]])

        bais = tf.get_variable("merge_bias", (1, 1, self.output_size))
        output += bais
        return get_keras_activation(self.activation)(output)


class LinearMerge(SequenceMapperWithContext):
    def __init__(self, n_out: int, use_bias: bool=True, init="glorot_uniform", activation="tanh"):
        self.n_out = n_out
        self.use_bias = use_bias
        self.init = init
        self.activation = activation

    def apply(self, is_train, x, c, mask=None, context_mask=None):
        c_w = tf.get_variable("context_weights", shape=(c.shape.as_list()[-1], self.n_out),
                              dtype=tf.float32, initializer=get_keras_initialization(self.init))
        c_projected = tf.matmul(c, c_w)

        x_w = tf.get_variable("input_weights", shape=(x.shape.as_list()[-1], self.n_out),
                              dtype=tf.float32, initializer=get_keras_initialization(self.init))
        x_proj = tf.tensordot(x, x_w, [[2], [0]])
        total = x_proj + tf.expand_dims(c_projected, 1)

        if self.use_bias:
            bias = tf.get_variable("bias", shape=self.n_out, dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            total += tf.expand_dims(tf.expand_dims(bias, 0), 0)

        return get_keras_activation(self.activation)(total)


class DropoutLayer(Updater):
    def __init__(self, keep_probs: float):
        self.keep_prob = keep_probs

    def apply(self, is_train, x, mask=None):
        return dropout(x, self.keep_prob, is_train)


class VariationalDropoutLayer(SequenceMapper):
    """
    `VariationalDropout` is an overload term, but this is in particular referring to
    https://arxiv.org/pdf/1506.02557.pdf were the dropout mask is consistent across the time dimension
    """

    def __init__(self, keep_prob: float):
        self.keep_prob = keep_prob

    def apply(self, is_train, x, mask=None):
        shape = tf.shape(x)
        return dropout(x, self.keep_prob, is_train, [shape[0], 1, shape[2]])


class FixedDropoutLayer(SequenceMapper):
    def __init__(self, keep_probs: float, axis: int):
        self.axis = axis
        self.keep_probs = keep_probs

    def apply(self, is_train, x, mask=None):
        shape = tf.shape(x)
        noise_shape = []
        for i in range(len(shape)):
            if i == self.axis:
                noise_shape.append(1)
            else:
                noise_shape.append(shape[i])
        return dropout(x, self.keep_probs, is_train, noise_shape)


class Conv1d(Mapper):
    def __init__(self, num_filters, filter_size, keep_probs, activation="relu"):
        self.keep_probs = keep_probs
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation

    def apply(self, is_train, x, mask=None):
        num_channels = x.get_shape()[3]
        filter_ = tf.get_variable("conv1d/filters", shape=[1, self.filter_size, num_channels, self.num_filters], dtype='float')
        bias = tf.get_variable("conv1d/bias", shape=[self.num_filters], dtype='float')
        strides = [1, 1, 1, 1]
        if self.keep_probs < 1.0:
            x = dropout(x, self.keep_probs, is_train)
        fn = get_keras_activation(self.activation)
        return fn(tf.nn.conv2d(x, filter_, strides, "VALID") + bias)


class MaxPool(Encoder):
    def __init__(self, map_layer: Optional[Mapper]=None, min_val: float=0):
        self.map_layer = map_layer
        self.min_val = min_val

    def apply(self, is_train, x, mask=None):
        if self.map_layer is not None:
            x = self.map_layer.apply(is_train, x, mask)

        rank = len(x.shape) - 2
        if mask is not None:
            shape = tf.shape(x)
            mask = tf.sequence_mask(tf.reshape(mask, (-1,)), shape[-2])
            mask = tf.cast(tf.reshape(mask, (shape[0], shape[1], shape[2], 1)), tf.float32)
            # this min_val thing is kind of a hack, really we should do something like compute the
            # min val over the entire batch, or maybe just pick a very negative values, or maybe
            # do something a bit more finicky with tf.bool_mask
            # In practice it doesn't seem to be problem, and some of the earlier models used these
            # scheme so I have been sticking with it.
            if self.min_val == 0:
                x *= mask
            else:
                x = x * mask + self.min_val * (1 - mask)
            return tf.maximum(tf.reduce_max(x, axis=rank), tf.fill([1] * (len(x.shape)-1),
                                                                   float(self.min_val)))
        else:
            return tf.reduce_max(x, axis=rank)

    def __setstate__(self, state):
        if "min_val" not in state:
            state["min_val"] = 0
        if "state" in state:
            if "min_val" not in state["state"]:
                state["state"]["min_val"] = 0
        super().__setstate__(state)


class ReduceSequenceLayer(SequenceEncoder):
    def __init__(self, reduce: str, apply_mask=True):
        self.reduce = reduce
        self.apply_mask = apply_mask

    def apply(self, is_train, x, mask=None):
        if mask is not None:
            answer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)
            if self.apply_mask:
                x *= answer_mask
        else:
            answer_mask = None

        if self.reduce == "max":
            if mask is not None:
                raise NotImplementedError()
            return tf.reduce_max(x, axis=1)
        elif self.reduce == "mean":
            if mask is not None:
                return tf.reduce_sum(x * answer_mask, axis=1) / tf.cast(tf.expand_dims(mask, 1), tf.float32)
            else:
                return tf.reduce_mean(x, axis=1)
        elif self.reduce == "sum":
            if mask is not None:
                return tf.reduce_sum(x * answer_mask, axis=1)
            else:
                return tf.reduce_sum(x, axis=1)
        else:
            raise ValueError()


class ChainConcat(SequenceBiMapper):
    """ How bidaf builds start/end logits """

    def __init__(self, start_layer: SequenceMapper, end_layer: SequenceMapper,
                 soft_select_start_word: bool=True, use_original: bool=True,
                 use_start_layer: bool=True, init: str="glorot_uniform"):
        self.init = init
        self.use_original = use_original
        self.start_layer = start_layer
        self.use_start_layer = use_start_layer
        self.end_layer = end_layer
        self.soft_select_start_word = soft_select_start_word

    def apply(self, is_train, context_embed, context_mask=None):
        init_fn = get_keras_initialization(self.init)
        with tf.variable_scope("start_layer"):
            m1 = self.start_layer.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(tf.concat([m1, context_embed], axis=2), 1,
                                      activation=None, kernel_initializer=init_fn)
            masked_logits1 = exp_mask(tf.squeeze(logits1, squeeze_dims=[2]), context_mask)
            prediction1 = tf.nn.softmax(masked_logits1)

        m2_input = []
        if self.use_original:
            m2_input.append(context_embed)
        if self.use_start_layer:
            m2_input.append(m1)
        if self.soft_select_start_word:
            soft_select = tf.einsum("ai,aik->ak", prediction1, m1)
            soft_select_tiled = tf.tile(tf.expand_dims(soft_select, axis=1), [1, tf.shape(m1)[1], 1])
            m2_input += [soft_select_tiled, soft_select_tiled * m1]

        with tf.variable_scope("end_layer"):
            m2 = self.end_layer.apply(is_train, tf.concat(m2_input, axis=2), context_mask)

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(tf.concat([m2, context_embed], axis=2), 1,
                                      activation=None, kernel_initializer=init_fn)

        return logits1, logits2

    def __setstate__(self, state):
        if "state" in state:
            if "aggregate" not in state["state"]:
                state["state"]["aggregate"] = None
        return super().__setstate__(state)


class ReduceLayer(Encoder):
    def __init__(self, reduce: str, map_layer: Optional[Mapper]=None, mask=True):
        self.map_layer = map_layer
        self.reduce = reduce
        self.mask = mask

    def apply(self, is_train, x, mask=None):
        if not self.mask:
            # Ignore the mask, this option exists since BiDaF does not account for masking here as well
            mask = None

        if mask is not None:
            valid_mask = tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32)
            for i in range(len(x.shape) - 2):
                tf.expand_dims(valid_mask, len(valid_mask.shape))
        else:
            valid_mask = None

        if self.map_layer is not None:
            x = self.map_layer.apply(is_train, x, mask)
        rank = len(x.shape) - 2

        if self.reduce == "max":
            if mask is not None:
                raise NotImplementedError()
            else:
                return tf.reduce_max(x, axis=rank)
        elif self.reduce == "mean":
            if valid_mask is not None:
                x *= valid_mask
                return tf.reduce_sum(x, axis=rank) / tf.cast(tf.expand_dims(mask, 1), tf.float32)
            else:
                return tf.reduce_mean(x, axis=rank)
        elif self.reduce == "sum":
            if valid_mask is not None:
                x *= valid_mask
            return tf.reduce_sum(x, axis=rank)
        else:
            raise ValueError()

    def __setstate__(self, state):
        if "state" in state:
            if "mask" not in state["state"]:
                state["state"]["mask"] = False
        return super().__setstate__(state)


class WithProduct(FixedMergeLayer):
    def apply(self, is_train, tensor1, tensor2, mask) -> tf.Tensor:
        return tf.concat([tensor1, tf.expand_dims(tensor2, 1) * tensor1], axis=2)


class WithTiled(FixedMergeLayer):
    def apply(self, is_train, tensor1, tensor2, mask) -> tf.Tensor:
        tiled = tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor1)[1], 1])
        return tf.concat([tensor1, tiled], axis=2)


class WithProjectedProduct(FixedMergeLayer):
    def __init__(self, init="glorot_uniform", include_tiled=False):
        self.init = init
        self.include_tiled = include_tiled

    def apply(self, is_train, tensor1, tensor2, mask) -> tf.Tensor:
        context_size = tensor2.shape.as_list()[-1]
        project_w = tf.get_variable("project_w", (tensor1.shape.as_list()[-1], context_size))
        projected = tf.tensordot(tensor1, project_w, axes=[[2], [0]])
        out = [tensor1, projected * tf.expand_dims(tensor2, 1)]
        if self.include_tiled:
            out.append(tf.tile(tf.expand_dims(tensor2, 1), [1, tf.shape(tensor1)[1], 1]))

        return tf.concat(out, axis=2)