from typing import Optional

import tensorflow as tf
from docqa.nn.layers import get_keras_activation, get_keras_initialization, SequenceMapper, SequenceEncoder, \
    MergeLayer, Encoder
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnCompatibleGRUCell
from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMBlockFusedCell, GRUBlockCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn

from docqa.configurable import Configurable
from docqa.nn.ops import dropout


class _CudnnRnn(Configurable):
    """
    Base class for using Cudnn's RNNs methods. Tensorflow's API for Cudnn is a bit gnarly,
    so this is isn't pretty.
    """

    def __init__(self,
                 kind: str,
                 n_units,
                 n_layers=1,
                 # Its not obvious how to compute fan_in/fan_out for these models
                 # so we recommend avoiding glorot initialization for now
                 w_init=TruncatedNormal(stddev=0.05),
                 recurrent_init=None,
                 bidirectional=True,
                 learn_initial_states: bool=False,
                 lstm_bias=1,
                 keep_recurrent: float=1):
        if bidirectional is None or n_layers is None or n_units is None:
            raise ValueError()
        if kind not in ["GRU", "LSTM"]:
            raise ValueError()
        self._kind = kind
        self.keep_recurrent = keep_recurrent
        self.lstm_bias = lstm_bias
        self.n_units = n_units
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.w_init = w_init
        self.recurrent_init = recurrent_init
        self.learn_initial_states = learn_initial_states

    def _apply_transposed(self, is_train, x):
        w_init = get_keras_initialization(self.w_init)
        r_init = None if self.recurrent_init is None else get_keras_initialization(self.recurrent_init)
        x_size = x.shape.as_list()[-1]
        if x_size is None:
            raise ValueError("Last dimension must be defined (have shape %s)" % str(x.shape))

        if self._kind == "GRU":
            cell = cudnn_rnn_ops.CudnnGRU(self.n_layers, self.n_units, x_size, input_mode="linear_input")
        elif self._kind == "LSTM":
            cell = cudnn_rnn_ops.CudnnLSTM(self.n_layers, self.n_units, x_size, input_mode="linear_input")
        else:
            raise ValueError()

        n_params = cell.params_size().eval()
        weights, biases = cell.params_to_canonical(tf.zeros([n_params]))

        def init(shape, dtype=None, partition_info=None):
            # This a bit hacky, since the api for these models is akward. We have to compute the shape of
            # the weights / biases by calling `cell.params_to_canonical` with a unused tensor, and then
            # use .eval() to actually get the shape. Then we can apply the user-requested initialzers
            if self._kind == "LSTM":
                is_recurrent = [False, False, False, False, True, True, True, True]
                is_forget_bias = [False, True, False, False, False, True, False, False]
            else:
                is_recurrent = [False, False, False, True, True, True]
                is_forget_bias = [False] * 6

            init_biases = [tf.constant(self.lstm_bias/2.0, tf.float32, (self.n_units,)) if z else tf.zeros(self.n_units)
                           for z in is_forget_bias]
            init_weights = []

            for w, r in zip(weights, is_recurrent):
                if r and r_init is not None:
                    init_weights.append(tf.reshape(r_init((self.n_units, self.n_units), w.dtype), tf.shape(w)))
                else:
                    init_weights.append(w_init(tf.shape(w).eval(), w.dtype))
            out = cell.canonical_to_params(init_weights, init_biases)
            out.set_shape((n_params, ))

            return out

        parameters = tf.get_variable(
            "gru_parameters",
            n_params,
            tf.float32,
            initializer=init
        )

        if self.keep_recurrent < 1:
            # Not super well test, try to figure out which indices in `parameters` are recurrent weights and drop them
            # this is implementing drop-connect for the recurrent weights
            is_recurrent = weights[:len(weights) // 2] + [tf.ones_like(w) for w in weights[len(weights) // 2:]]
            recurrent_mask = cell.canonical_to_params(is_recurrent, biases)  # ones at recurrent weights
            recurrent_mask = 1 - recurrent_mask * (1 - self.keep_recurrent)  # ones are non-recurrent param, keep_prob elsewhere
            parameters = tf.cond(is_train,
                                 lambda: tf.floor(tf.random_uniform((n_params, )) + recurrent_mask) * parameters,
                                 lambda: parameters)

        if self._kind == "LSTM":
            if self.learn_initial_states:
                raise NotImplementedError()
            else:
                initial_state_h = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
                initial_state_c = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
            out = cell(x, initial_state_h, initial_state_c, parameters, True)
        else:
            if self.learn_initial_states:
                initial_state = tf.get_variable("initial_state", self.n_units,
                                                tf.float32, tf.zeros_initializer())
                initial_state = tf.tile(tf.expand_dims(tf.expand_dims(initial_state, 0), 0),
                                        [self.n_layers, tf.shape(x)[1], 1])
            else:
                initial_state = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
            out = cell(x, initial_state, parameters, True)
        return out


class CudnnRnnMapper(_CudnnRnn):
    def map(self, is_train, x, mask=None):
        x = tf.transpose(x, [1, 0, 2])

        if self.bidirectional:
            with tf.variable_scope("forward"):
                fw = self._apply_transposed(is_train, x)[0]
            with tf.variable_scope("backward"):
                bw = self._apply_transposed(is_train, tf.reverse_sequence(x, mask, 0, 1))[0]
                bw = tf.reverse_sequence(bw, mask, 0, 1)
            out = tf.concat([fw, bw], axis=2)
        else:
            out = self._apply_transposed(is_train, x)[0]
        out = tf.transpose(out, [1, 0, 2])
        if mask is not None:
            out *= tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(out)[1]), tf.float32), 2)
        return out


class CudnnGru(CudnnRnnMapper, SequenceMapper):
    def __init__(self,
                 n_units,
                 n_layers=1,
                 keep_recurrent=1,
                 w_init=TruncatedNormal(stddev=0.05),
                 recurrent_init=None,
                 bidirectional=True,
                 learn_initial_states=False):
        super().__init__("GRU", n_units, n_layers, w_init, recurrent_init, bidirectional,
                         learn_initial_states, 1, keep_recurrent)

    def apply(self, is_train, x, mask=None):
        return super().map(is_train, x, mask)

    def __setstate__(self, state):
        if "state" in state:
            if "_kind" not in state["state"]:
                state["state"]["_kind"] = "GRU"
            if "learn_initial_states" not in state["state"]:
                state["state"]["learn_initial_states"] = False
            if "recurrent_init" not in state["state"]:
                state["state"]["recurrent_init"] = None
            if "keep_recurrent" not in state["state"]:
                state["state"]["keep_recurrent"] = 1
        super().__setstate__(state)


class CudnnLstm(CudnnRnnMapper, SequenceMapper):
    def __init__(self,
                 n_units,
                 n_layers=1,
                 lstm_bias=1,
                 w_init=TruncatedNormal(stddev=0.05),
                 recurrent_init=None,
                 bidirectional=True,
                 learn_initial_states=False):
        super().__init__("LSTM", n_units, n_layers, w_init, recurrent_init, bidirectional,
                         learn_initial_states, lstm_bias)

    def apply(self, is_train, x, mask=None):
        return super().map(is_train, x, mask)

    def __setstate__(self, state):
        if "state" in state:
            if "recurrent_init" not in state["state"]:
                state["state"]["recurrent_init"] = None
            if "keep_recurrent" not in state["state"]:
                state["state"]["keep_recurrent"] = 1
        super().__setstate__(state)


class FusedRecurrentEncoder(SequenceEncoder):
    """ Use `LSTMBlockFusedCell` and return the last hidden states """
    def __init__(self, n_units, hidden=True, state=False):
        self.n_units = n_units
        self.hidden = hidden
        self.state = state

    def apply(self, is_train, x, mask=None):
        x = tf.transpose(x, [1, 0, 2])  # to time first
        state = LSTMBlockFusedCell(self.n_units)(x, dtype=tf.float32, sequence_length=mask)[1]
        if self.state and self.hidden:
            state = tf.concat(state, 1)
        elif self.hidden:
            state = state.h
        elif self.state:
            state = state.c
        else:
            raise ValueError()
        return state

    def __setstate__(self, state):
        if "n_units" not in state:
            self.__dict__ = state["state"]
        else:
            self.__dict__ = state


class BiDirectionalFusedLstm(SequenceMapper):
    """ Use `LSTMBlockFusedCell` and return all hidden states """

    def __init__(self, n_units, use_peepholes=False):
        self.n_units = n_units
        self.use_peepholes = use_peepholes

    def apply(self, is_train, inputs, mask=None):
        inputs = tf.transpose(inputs, [1, 0, 2])  # to time first
        with tf.variable_scope("forward"):
            cell = LSTMBlockFusedCell(self.n_units, use_peephole=self.use_peepholes)
            fw = cell(inputs, dtype=tf.float32, sequence_length=mask)[0]
        with tf.variable_scope("backward"):
            cell = LSTMBlockFusedCell(self.n_units, use_peephole=self.use_peepholes)
            inputs = tf.reverse_sequence(inputs, mask, seq_axis=0, batch_axis=1)
            bw = cell(inputs, dtype=tf.float32, sequence_length=mask)[0]
            bw = tf.reverse_sequence(bw, mask, seq_axis=0, batch_axis=1)
        out = tf.concat([fw, bw], axis=2)
        out = tf.transpose(out, [1, 0, 2])  # back to batch first
        return out


class EncodeOverTime(Encoder):
    def __init__(self, enc: SequenceEncoder, mask=True):
        self.enc = enc
        self.mask = mask

    def apply(self, is_train, x, mask=None):
        lst = x.shape.as_list()
        batch = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1, lst[-2], lst[-1]])  # (batch*words, char, dim)

        if not self.mask:
            mask = None
        if mask is not None:
            mask = tf.reshape(mask, (-1,))

        encoding = self.enc.apply(is_train, flattened, mask)

        # reshape to the original size
        enc = tf.reshape(encoding, [batch, -1, encoding.shape.as_list()[-1]])
        return enc

""""
Everything below here is for using the the tensorflow's rnn module, I generally have abandoning using this
stuff since the FusedCell and Cudnn classes are 5-10x faster
"""


def _linear(args, output_size, bias, bias_init, weight_init):
    dtype = args.dtype

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
            "weights", [args.shape.as_list()[-1], output_size], dtype=dtype, initializer=weight_init)
        res = tf.matmul(args, weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=bias_init)
        return tf.nn.bias_add(res, biases)


def _compute_gates(input, hidden, num_units, forget_bias, kernel_init, recurrent_init,
                   return_matrix=False):
    # Have to stack the weight initlaizers carefully, so that the initializer work as intended
    # for kernel/recurrent
    num_inputs = input.shape.as_list()[1]
    def _init_stacked_weights(shape, dtype=None, partition_info=None):
        if partition_info is not None:
            raise ValueError()
        kernal_shape = list(shape)
        kernal_shape[0] = num_inputs
        recurrent_shape = list(shape)
        recurrent_shape[0] -= num_inputs
        return tf.concat([kernel_init(kernal_shape, dtype), recurrent_init(recurrent_shape, dtype)], axis=0)

    def _init_stacked_bias(shape, dtype=None, partition_info=None):
        if partition_info is not None:
            raise ValueError()
        if shape[0] != num_units*4:
            raise ValueError()
        shape = list(shape)
        shape[0] = num_units
        inits = [tf.zeros_initializer(), tf.zeros_initializer(), tf.constant_initializer(forget_bias), tf.zeros_initializer()]
        return tf.concat([i(shape, dtype) for i in inits], axis=0)

    mat = _linear(tf.concat([input, hidden], axis=1), 4*num_units, True,
                   bias_init=_init_stacked_bias,
                   weight_init=_init_stacked_weights)
    if return_matrix:
        return mat
    else:
        return tf.split(value=mat, num_or_size_splits=4, axis=1)


try:
    # For tf 1.1
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
except ImportError:
    # for tf 1.2 (and hopefully up)
    from tensorflow.python.ops.rnn_cell_impl import RNNCell


class InitializedLSTMCell(RNNCell):
    def __init__(self, num_units,
                 kernel_initializer,
                 recurrent_initializer,
                 activation,
                 recurrent_activation,
                 forget_bias=1.0,
                 keep_recurrent_probs=1.0,
                 is_train=None,
                 scope=None):
        self.scope = scope if scope is not None else "init_lstm_cell"
        self.is_train = is_train
        self.num_units = num_units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.forget_bias = forget_bias
        self.keep_recurrent_probs = keep_recurrent_probs

    @property
    def state_size(self):
        return LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(self.scope):
            c, h = state
            h = dropout(h, self.keep_recurrent_probs, self.is_train)

            mat = _compute_gates(inputs, h, self.num_units, self.forget_bias,
                                        self.kernel_initializer, self.recurrent_initializer, True)

            i, j, f, o = tf.split(value=mat, num_or_size_splits=4, axis=1)

            new_c = (c * self.recurrent_activation(f) + self.recurrent_activation(i) *
                     self.activation(j))
            new_h = self.activation(new_c) * self.recurrent_activation(o)

            new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state


class GRUCell(RNNCell):

    def __init__(self, num_units, bias_init, kernel_init, recurrent_init,
                 candidate_init, activation=tf.tanh):
        self.num_units = num_units
        self.activation = activation
        self.kernal_init = kernel_init
        self.recurrent_init = recurrent_init
        self.bias_init = bias_init
        self.candidate_init = candidate_init

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("gru"):

            def _init_stacked_weights(shape, dtype=None, partition_info=None):
                kernal_shape = list(shape)
                kernal_shape[0] = inputs.shape.as_list()[-1]
                recurrent_shape = list(shape)
                recurrent_shape[0] = state.shape.as_list()[-1]
                return tf.concat([self.kernal_init(kernal_shape, dtype), self.recurrent_init(recurrent_shape, dtype)], axis=0)

            value = tf.sigmoid(_linear(tf.concat([inputs, state], axis=1), self.num_units*2,
                                       True, self.bias_init, _init_stacked_weights))
            r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        with tf.variable_scope("candidate"):
            c = self.activation(_linear(tf.concat([inputs, r * state], axis=1), self.num_units, True,
                                        tf.zeros_initializer(), self.candidate_init))
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class RnnCellSpec(Configurable):
    """ Configurable specification for a RNN cell. RNNCell is stateful at least in 1.1 due to
    their scope-saving thing, but layers are not supposed to be stateful we build a new RNNCell
    each call. """
    def convert_to_state(self, variables):
        raise ValueError()

    def build_initial_state_var(self, batch_size, cell):
        initial = []
        tile_arr = tf.constant([batch_size, 1], dtype=tf.int32)

        state_size = cell.state_size
        for i, s in enumerate(state_size):
            if hasattr(state_size, "_fields"):
                name = "initial_state_%s" % state_size._fields[i]
            else:
                name = "initial_state_%d" % i
            var = tf.get_variable(name, s, tf.float32)
            initial.append(tf.tile(tf.expand_dims(var, 0), tile_arr))
        return self.convert_to_state(initial)

    def __call__(self, is_train, scope=None):
        raise NotImplementedError()


class LstmCellSpec(RnnCellSpec):
    """
    Build a LSTM cell with custom initialization
    """

    def __init__(self, num_units,
                 forget_bias=1.0,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 activation='tanh',
                 recurrent_activation="sigmoid",
                 keep_recurrent_probs=1):
        self.keep_recurrent_probs = keep_recurrent_probs
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer

    def convert_to_state(self, variables):
        if len(variables) != 2:
            raise ValueError()
        return LSTMStateTuple(variables[0], variables[1])

    def __call__(self, is_train, scope=None):
        activation = get_keras_activation(self.activation)
        recurrent_activation = get_keras_activation(self.recurrent_activation)
        kernel_initializer = get_keras_initialization(self.kernel_initializer)
        recurrent_initializer = get_keras_initialization(self.recurrent_initializer)
        if activation is None or kernel_initializer is None \
                or recurrent_initializer is None or recurrent_activation is None:
            raise ValueError()

        cell = InitializedLSTMCell(self.num_units, kernel_initializer,
                                   recurrent_initializer, activation,
                                   recurrent_activation, self.forget_bias,
                                   self.keep_recurrent_probs, is_train, scope)
        return cell

    def __setstate__(self, state):
        if "state" in state:
            if "keep_recurrent_probs" not in state["state"]:
                state["state"]["keep_recurrent_probs"] = 1.0
        super().__setstate__(state)


class BlockGruCellSpec(RnnCellSpec):

    def __init__(self, num_units):
        self.num_units = num_units

    def convert_to_state(self, variables):
        if len(variables) != 2:
            raise ValueError()
        return LSTMStateTuple(variables[0], variables[1])

    def __call__(self, is_train, scope=None):
        return GRUBlockCell(self.num_units)


class GruCellSpec(RnnCellSpec):

    def __init__(self, num_units,
                 bais_init=1.0,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 candidate_initializer='glorot_uniform',
                 activation='tanh'):
        self.num_units = num_units
        self.bais_init = bais_init
        self.activation = activation
        self.candidate_initializer = candidate_initializer
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer

    def convert_to_state(self, variables):
        if len(variables) != 2:
            raise ValueError()
        return LSTMStateTuple(variables[0], variables[1])

    def __call__(self, is_train, scope=None):
        activation = get_keras_activation(self.activation)
        recurrent_initializer = get_keras_initialization(self.recurrent_initializer)
        kernel_initializer = get_keras_initialization(self.kernel_initializer)
        candidate_initializer = get_keras_initialization(self.candidate_initializer)
        return GRUCell(self.num_units, tf.constant_initializer(self.bais_init),
                       kernel_initializer, recurrent_initializer, candidate_initializer, activation)


class CompatGruCellSpec(RnnCellSpec):

    def __init__(self, num_units):
        self.num_units = num_units

    def __call__(self, is_train, scope=None):
        return CudnnCompatibleGRUCell(self.num_units)


class RecurrentEncoder(SequenceEncoder):
    def __init__(self, cell_spec, output):
        self.cell_spec = cell_spec
        self.output = output

    def apply(self, is_train, x, mask=None):
        state = dynamic_rnn(self.cell_spec(is_train), x, mask, dtype=tf.float32)[1]
        if isinstance(self.output, int):
            return state[self.output]
        else:
            if self.output is None:
                if not isinstance(state, tf.Tensor):
                    raise ValueError()
                return state
            for i,x in enumerate(state._fields):
                if x == self.output:
                    return state[i]
            raise ValueError()


class BiRecurrentEncoder(SequenceEncoder):
    def __init__(self, cell_spec, output, merge: Optional[MergeLayer]=None):
        self.cell_spec = cell_spec
        self.output = output
        self.merge = merge

    def apply(self, is_train, x, mask=None):
        states = bidirectional_dynamic_rnn(self.cell_spec(is_train), self.cell_spec(is_train), x, mask, dtype=tf.float32)[1]
        output = []
        for state in states:
            for i,x in enumerate(state._fields):
                if x == self.output:
                    output.append(state[i])
        if self.merge is not None:
            return self.merge.apply(is_train, output[0], output[1])
        else:
            return tf.concat(output, axis=1)


class RecurrentMapper(SequenceMapper):

    def __init__(self, cell_spec, learn_initial=False):
        self.cell_spec = cell_spec
        self.learn_initial = learn_initial

    def apply(self, is_train, inputs, mask=None):
        cell = self.cell_spec(is_train)
        batch_size = inputs.shape.as_list()[0]

        if self.learn_initial:
            initial = self.cell_spec.build_initial_state_var(batch_size, cell)
        else:
            initial = None

        return dynamic_rnn(cell, inputs, mask, initial, dtype=tf.float32)[0]


class BiRecurrentMapper(SequenceMapper):
    def __init__(self, fw, bw=None, merge: MergeLayer = None, swap_memory=False):
        self.fw = fw
        self.swap_memory = swap_memory
        self.bw = bw
        self.merge = merge

    def apply(self, is_train, inputs, mask=None):
        fw = self.fw(is_train)
        bw_spec = self.fw if self.bw is None else self.bw
        bw = bw_spec(is_train)

        if self.merge is None:
            return tf.concat(bidirectional_dynamic_rnn(fw, bw, inputs, mask, swap_memory=self.swap_memory,
                                                       dtype=tf.float32)[0], 2,)
        else:
            fw, bw = bidirectional_dynamic_rnn(fw, bw, inputs, mask,
                                               swap_memory=self.swap_memory, dtype=tf.float32)[0]
            return self.merge.apply(is_train, fw, bw)  # TODO this should be in a different scope


