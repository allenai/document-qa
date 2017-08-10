import tensorflow as tf
from typing import NamedTuple, Tuple, Optional

from tensorflow.contrib.layers import fully_connected, dropout
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import DynamicAttentionWrapper, BahdanauAttention
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from nn.layers import MergeLayer, Updater, AttentionMapper, GatingLayer, ConcatLayer, get_keras_initialization
from nn.recurrent_layers import RnnCellSpec
from nn.similarity_layers import SimilarityFunction


def _mask_memory(keys, memory, memory_sequence_length):
    if memory_sequence_length is not None:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen=tf.shape(memory)[1],
            dtype=memory.dtype)
        seq_len_mask = tf.expand_dims(seq_len_mask, 2)
        memory = memory * seq_len_mask
        if keys != memory:
            keys = keys * seq_len_mask
    return keys, memory


KeyValueAttentionState = NamedTuple("InputBasedAttentionState", [
    ("cell_state", Tuple),
    ("cell_output", tf.Tensor),
    ("attention", tf.Tensor)
])


WindowAttentionState = NamedTuple("WindowAttentionState", [
    ("cell_state", Tuple),
    ("cell_output", tf.Tensor),
    ("attention", tf.Tensor),
    ("ix", tf.Tensor)
])


class WeightedWindowAttention(RNNCell):
    def __init__(self,
                 memory,
                 memory_weights,
                 window_size: int,
                 cell: RNNCell):
        self.window_size = window_size
        self.cell = cell
        self.memory = memory
        self.memory_weights = memory_weights

    @property
    def state_size(self):
        return WindowAttentionState(self.cell.state_size, self.cell.output_size, self.memory.shape.as_list()[2], 1)

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, inputs, state: KeyValueAttentionState, scope=None):
        with tf.variable_scope(None, "InputBasedAttention"):
            ix = tf.cast(state.ix[0][0], tf.int32) + 1
            start = tf.maximum(ix-self.window_size, 0)
            probs = tf.nn.softmax(self.memory_weights[:, start:ix])

            context = tf.einsum("aij,ai->aj", self.memory[:, start:ix], probs)

            factors = tf.concat([inputs, state.cell_output, context], axis=1)

            cell_out, cell_state = self.cell(factors, state.cell_state)
            return cell_out, WindowAttentionState(cell_state, cell_out, context, state.ix + 1)


class KeyValueAttention(RNNCell):
    def __init__(self,
                 memory,
                 keys,
                 is_train,
                 cell: RNNCell,
                 init="glorot_uniform",
                 attention=SimilarityFunction,
                 combine: MergeLayer = ConcatLayer(),
                 gate_attention: bool=False,
                 memory_sequence_length=None,
                 use_cell_output=True,
                 use_cur_input=True,
                 scope=None):
        self.init = init
        self.gate_attention = gate_attention
        self.combine = combine
        self.is_train = is_train
        self.use_cur_input = use_cur_input
        self.use_cell_output = use_cell_output
        self.attention_key_size = keys.shape.as_list()[-1]
        self.cell = cell
        self.attention = attention
        self.scope = scope

        if not memory.get_shape()[2:].is_fully_defined():
            raise ValueError()

        keys, memory = _mask_memory(keys, memory, memory_sequence_length)

        # memory we will look up vectors on, these are the vectors we will attend to
        # shape (batch X n_memories X dim)
        self.memory = memory
        self.keys = keys

    @property
    def state_size(self):
        return KeyValueAttentionState(self.cell.state_size, self.cell.output_size, self.memory.shape.as_list()[2])

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, inputs, state: KeyValueAttentionState, scope=None):
        init = get_keras_initialization(self.init)

        with tf.variable_scope(None, "InputBasedAttention"):
            # concat the inputs to use as the part of the attention query
            atten_factors = []
            if self.use_cell_output:
                atten_factors.append(state.cell_output)
            if self.use_cur_input:
                atten_factors.append(inputs)
            atten_factors = tf.concat(atten_factors, axis=1)

            # (batch x n_keys)
            alignments = self.attention.get_one_sided_scores(atten_factors, self.keys)

            # batch matrix mult w/size (1xkey_dim) matrices to compute attention
            alignments = tf.expand_dims(alignments, 1)
            attention = tf.matmul(alignments, self.memory)
            attention = tf.squeeze(attention, [1])

            # If gated, another linear layer is applied to select the important parts of attention
            if self.gate_attention:
                atten_size = attention.shape.as_list()[-1]
                input_size = inputs.shape.as_list()[-1]
                gate_w = tf.get_variable("gate_w", (atten_size + input_size, atten_size), initializer=init)
                gate = tf.sigmoid(tf.matmul(tf.concat([inputs, attention], axis=1), gate_w))
                attention = gate * attention

            # Combine with the inputs
            cell_input = self.combine.apply(self.is_train, attention, inputs)

            # Run the underlying cell, finding it the inputs and the previous hidden state
            cell_out, cell_state = self.cell(cell_input, state.cell_state)
            print(cell_state)

            return cell_out, KeyValueAttentionState(cell_state, cell_out, attention)


class RecurrentAttention(AttentionMapper):
    def __init__(self,
                 cell_spec: RnnCellSpec,
                 attention=SimilarityFunction,
                 merge_layer: Optional[MergeLayer]=None,
                 gate_attention=False):
        self.merge_layer = merge_layer
        self.cell_spec = cell_spec
        self.attention = attention
        self.gate_attention = gate_attention

    def apply(self, is_train, x, keys, memory, x_mask=None, mem_mask=None):
        cell = KeyValueAttention(memory, keys, is_train,
                                     cell=self.cell_spec(is_train),
                                     combine=ConcatLayer(),
                                     attention=self.attention,
                                     gate_attention=self.gate_attention,
                                     memory_sequence_length=mem_mask)

        with tf.variable_scope("atten"):
            context_atten_embed, _ = bidirectional_dynamic_rnn(cell, cell,
                                                               dtype=tf.float32, inputs=x,
                                                               sequence_length=x_mask)
        if self.merge_layer is not None:
            with tf.variable_scope("merge"):
                return self.merge_layer.apply(is_train, context_atten_embed[0], context_atten_embed[0])
        else:
            return tf.concat(context_atten_embed, axis=2)
