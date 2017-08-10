from typing import Optional

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from configurable import Configurable
from nn.layers import get_keras_activation, get_keras_initialization, SequenceMapper, SqueezeLayer, SequenceEncoder, \
    MergeLayer

from tensorflow.contrib.rnn import LSTMStateTuple, DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as _tmp_linear


from nn.ops import dropout


class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
             seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                          for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state


class RnnCellSpec(Configurable):
    """ Specification for a RNN cell. """
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

    def __init__(self, num_units,
                 forget_bias=1.0,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 activation='tanh',
                 recurrent_activation="sigmoid",
                 keep_probs=1,
                 keep_recurrent_probs=1):
        self.keep_recurrent_probs = keep_recurrent_probs
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.keep_probs = keep_probs
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
        if self.keep_probs >= 1:
            return cell
        else:
            return SwitchableDropoutWrapper(cell, is_train, self.keep_probs)

    def __setstate__(self, state):
        if "keep_recurrent_probs" not in state["state"]:
            state["state"]["keep_recurrent_probs"] = 1.0
        super().__setstate__(state)


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
    # TODO actually, maybe we should run the initlaizer once per each gate, not just
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

    def __call__(self, is_train):
        activation = get_keras_activation(self.activation)
        recurrent_initializer = get_keras_initialization(self.recurrent_initializer)
        kernel_initializer = get_keras_initialization(self.kernel_initializer)
        candidate_initializer = get_keras_initialization(self.candidate_initializer)
        return GRUCell(self.num_units, tf.constant_initializer(self.bais_init),
                       kernel_initializer, recurrent_initializer, candidate_initializer, activation)


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


class EncodeOverTime(SqueezeLayer):
    def __init__(self, enc: SequenceEncoder, mask: bool=True):
        self.enc = enc
        self.mask = mask

    def apply(self, is_train, x, mask=None):
        lst = x.shape.as_list()
        batch = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1, lst[-2], lst[-1]])  # (batch*words, char, dim)

        if self.mask:
            # we assume that characters w/all exactly zero embeds should be masked
            # this risks making a mistake and masking real values if a char-embed just happends to be zero
            # but that should be extremely unlikely
            char_mask = tf.reduce_sum(tf.cast(tf.reduce_all(tf.not_equal(flattened, 0), axis=2), tf.int32), axis=1)
        else:
            char_mask = None

        encoding = self.enc.apply(is_train, flattened, char_mask)

        # reshape to the original size
        enc = tf.reshape(encoding, [batch, -1, encoding.shape.as_list()[-1]])
        return enc


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
            return self.merge.apply(is_train, fw, bw)

    def __setstate__(self, state):
        if "merge" not in state["state"]:
            state["state"]["merge"] = None
        if "swap_memory" not in state["state"]:
            state["state"]["swap_memory"] = False
        super().__setstate__(state)


    # mat = tf.matmul(tf.concat([inputs, h], axis=1), tf.concat([kernal_w, recurrent_w], axis=0)) + bias
    #
    # hidden_std = tf.sqrt(tf.matmul(tf.pow(inputs, 2), tf.pow(kernal_w, 2)))



class BiRecurrentMapperRegularized(SequenceMapper):
    def __init__(self, regularization, root_epsilon: Optional[float], fw: LstmCellSpec, merge: MergeLayer = None):
        self.regularization = regularization
        self.root_epsilon = root_epsilon
        self.fw = fw
        self.merge = merge

    def apply(self, is_train, inputs, mask=None):
        fw = self.fw(is_train, "cell")
        bw = self.fw(is_train, "cell")
        fw.tmp_reg = True
        bw.tmp_reg = True
        batch_size = tf.shape(inputs)[0]
        fw_init = fw.zero_state(batch_size, tf.float32)
        bw_init = fw.zero_state(batch_size, tf.float32)
        input_size = inputs.shape.as_list()[-1]

        states = bidirectional_dynamic_rnn(fw, bw, inputs, mask, initial_state_fw=fw_init,
                                           scope="regulated_bi_rnn",
                                           initial_state_bw=bw_init, dtype=tf.float32)[0]
        with tf.variable_scope("regulated_bi_rnn/fw/cell/weights"):
            fw_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            if len(fw_w) != 1:
                print(fw_w)
                raise ValueError()
            fw_w = fw_w[0][:input_size, :]

        with tf.variable_scope("regulated_bi_rnn/bw/cell/weights"):
            bw_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            if len(bw_w) != 1:
                raise ValueError()
            bw_w = bw_w[0][:input_size, :]

        states = list(states)
        for w in [fw_w, bw_w]:

            # Flatten the time/word dimension and apply mask
            flat_inputs = tf.reshape(inputs, (-1, inputs.shape.as_list()[-1]))
            flat_inputs = tf.boolean_mask(flat_inputs, tf.reshape(tf.sequence_mask(mask), (-1,)))

            var = tf.matmul(tf.square(flat_inputs), tf.square(w))
            std = tf.sqrt(tf.clip_by_value(var, 1.0E-6, 1000))
            tf.add_to_collection("OutputVar", tf.reduce_mean(var))

            abs_input = tf.abs(flat_inputs)

            # Mean input per batch
            mean_abs_inputs = tf.reduce_mean(abs_input, axis=1)

            outputs = tf.tensordot(flat_inputs, w, [[1], [0]])

            flat_inputs_tile = tf.tile(tf.expand_dims(flat_inputs, 0), [15, 1, 1])
            dropout_error = tf.tensordot(tf.nn.dropout(flat_inputs_tile, 0.8), w, [[2], [0]]) - tf.expand_dims(outputs, 0)
            dropout_error = tf.abs(dropout_error)
            std_error = tf.random_normal([15, tf.shape(outputs)[0], tf.shape(outputs)[1]],
                                         0,
                                         tf.tile(tf.expand_dims(std*0.5, 0), [15, 1, 1]))
            std_error = tf.abs(std_error)
            tf.add_to_collection("DropError", tf.reduce_mean(dropout_error))
            tf.add_to_collection("StdError", tf.reduce_mean(std_error))
            tf.add_to_collection("ErrorDiff", tf.reduce_mean(dropout_error/std_error))

            abs_w = tf.abs(w)

            # Mean weight per output
            mean_abs_w = tf.reduce_mean(abs_w, axis=0)

            tf.add_to_collection("OutputL1", tf.reduce_mean(tf.abs(outputs)))
            tf.add_to_collection("OutputL2", tf.reduce_mean(tf.square(outputs)))
            tf.add_to_collection("OutputMean", tf.reduce_mean(outputs))

            # n_out = tf.cast(tf.shape(w)[1], tf.float32)
            # even_w = tf.reduce_mean(w, axis=1)
            # even_i = tf.reduce_mean(flat_inputs, axis=0)
            # tf.add_to_collection("EvenW", tf.reduce_mean(tf.tensordot(flat_inputs, even_w, [[1], [0]])))
            # tf.add_to_collection("EvenI", tf.reduce_mean(tf.tensordot(even_i, w, [[0], [0]])))
            # tf.add_to_collection("EvenB", tf.reduce_mean((tf.expand_dims(even_i, 1) * tf.expand_dims(even_w, 0))))


            sq_w = tf.square(w)
            tf.add_to_collection("Mean", tf.reduce_mean(abs_w))
            tf.add_to_collection("L1", tf.reduce_mean(abs_w))
            tf.add_to_collection("L2", tf.reduce_mean(sq_w))
            diff = abs_w - tf.expand_dims(mean_abs_w, 0)
            tf.add_to_collection("L1Deviation", tf.reduce_mean(tf.abs(diff)/tf.expand_dims(mean_abs_w, 0)))

            tf.add_to_collection("InputMean", tf.reduce_mean(flat_inputs))
            tf.add_to_collection("InputL1", tf.reduce_mean(abs_input))
            tf.add_to_collection("InputL2", tf.reduce_mean(tf.square(flat_inputs)))
            tf.add_to_collection("InputL1Deviation", tf.reduce_mean((abs_input - tf.expand_dims(mean_abs_inputs, 1))/tf.expand_dims(mean_abs_inputs, 1)))

            cov =  tf.matmul(abs_input - tf.expand_dims(mean_abs_inputs, 1),
                             abs_w - tf.expand_dims(mean_abs_w, 0), name="covaraince")

            # st. dev. between inputs for each batch
            std_inputs = tf.sqrt(tf.reduce_sum(tf.square(abs_input - tf.expand_dims(mean_abs_inputs, 1)), axis=1))

            # st. dev. between inputs weights for each each output
            std_w = tf.sqrt(tf.reduce_sum(tf.square(abs_w - tf.expand_dims(mean_abs_w, 0)), axis=0))

            # Normalized the cov to get the pearson correlation per each batch, output
            cov /= (tf.expand_dims(std_inputs, 1) * tf.expand_dims(std_w, 0) + 1.0E-7)
            tf.add_to_collection("WeightInputCorrelation", tf.reduce_mean(tf.abs(cov)))

        if self.merge is None:
            return tf.concat(states, 2)
        else:
            return self.merge.apply(is_train, states[0], states[1])

    def __setstate__(self, state):
        if "merge" not in state["state"]:
            state["state"]["merge"] = None
        super().__setstate__(state)


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




# class LearnedDropoutLSTMCellSpec(RnnCellSpec):
#     def __init__(self, num_units,
#                  keep_probs,
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  activation='tanh',
#                  recurrent_activation="sigmoid",
#                  forget_bias=1.0,
#                  aux_metric=None):
#         self.keep_probs = keep_probs
#         self.num_units = num_units
#         self.activation = activation
#         self.recurrent_activation = recurrent_activation
#         self.kernel_initializer = kernel_initializer
#         self.recurrent_initializer = recurrent_initializer
#         self.forget_bias = forget_bias
#         self.aux_metric = aux_metric
#
#     def __call__(self, is_train):
#         return LearnedDropoutLSTMCell(self.num_units, self.kernel_initializer,
#                                       self.recurrent_initializer, self.activation,
#                                       self.keep_probs, self.recurrent_activation,
#                                       self.forget_bias, is_train, self.aux_metric)
#
#     def convert_to_state(self, variables):
#         if len(variables) != 2:
#             raise ValueError()
#         return LSTMStateTuple(variables[0], variables[1])
#
#
# class LearnedDropoutLSTMCell(RNNCell):
#     def __init__(self, num_units,
#                  kernel_initializer,
#                  recurrent_initializer,
#                  activation,
#                  keep_probs,
#                  recurrent_activation,
#                  forget_bias=1.0,
#                  is_train=None,
#                  aux_metric=None):
#         self.keep_probs = keep_probs
#         self.is_train = is_train
#         self.num_units = num_units
#         self.activation = activation
#         self.recurrent_activation = recurrent_activation
#         self.kernel_initializer = kernel_initializer
#         self.recurrent_initializer = recurrent_initializer
#         self.forget_bias = forget_bias
#         self.aux_metric = aux_metric
#
#     @property
#     def state_size(self):
#         return LSTMStateTuple(self.num_units, self.num_units)
#
#     @property
#     def output_size(self):
#         return self.num_units
#
#     def __call__(self, inputs, state, scope=None):
#         with tf.variable_scope("learned_dropout_lstm_cell"):
#             c, h = state
#             i_dim = inputs.shape.as_list()[-1]
#
#             random_tensor = self.keep_probs + tf.random_uniform(tf.shape(inputs), minval=0, maxval=1, dtype=inputs.dtype)
#             binary_tensor = tf.floor(random_tensor)
#             drop_i = inputs * binary_tensor
#
#             if self.aux_metric:
#                 fill_in = fully_connected(tf.stop_gradient(tf.concat([inputs, h], axis=1)), i_dim, activation_fn=None)
#                 if self.aux_metric == "l2":
#                     reg = tf.reduce_mean(tf.pow(tf.stop_gradient(inputs) - fill_in, 2))
#                 elif self.aux_metric == "l1":
#                     reg = tf.reduce_mean(tf.abs(tf.stop_gradient(inputs) - fill_in))
#                 else:
#                     raise ValueError()
#                 tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)
#                 fill_in = tf.stop_gradient(fill_in)
#             else:
#                 fill_in = fully_connected(tf.concat([inputs, h], axis=1), i_dim, activation_fn=None)
#
#             inputs = tf.cond(self.is_train, lambda: drop_i + fill_in * (1 - binary_tensor), lambda: inputs)
#
#             k_init = get_keras_initialization(self.kernel_initializer)
#             r_init = get_keras_initialization(self.recurrent_initializer)
#             i, j, f, o = _compute_gates(inputs, h, self.num_units, self.forget_bias, k_init, r_init)
#
#             r_activation = get_keras_activation(self.recurrent_activation)
#             activation = get_keras_activation(self.activation)
#             new_c = (c * r_activation(f) + r_activation(i) * activation(j))
#             new_h = activation(new_c) * r_activation(o)
#
#             new_state = LSTMStateTuple(new_c, new_h)
#
#         return new_h, new_state