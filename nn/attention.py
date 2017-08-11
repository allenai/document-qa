from typing import Optional

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

from configurable import Configurable
from nn.layers import AttentionMapper, MergeLayer, SequenceEncoder, get_keras_initialization, SequenceMapper, \
    SqueezeLayer, get_keras_activation, Mapper, SequenceMultiEncoder
from nn.ops import VERY_NEGATIVE_NUMBER, exp_mask
from nn.similarity_layers import SimilarityFunction, compute_attention_mask
import numpy as np

"""
Module for non-recurrent attention layeres
"""


class PaddedStaticAttention(AttentionMapper):

    def __init__(self, attention: SimilarityFunction,
                 n_extra_keys: int, learn_extra_mem: int,
                 bias: bool, merge: Optional[MergeLayer]=None):
        self.attention = attention
        self.merge = merge
        self.n_extra_keys = n_extra_keys
        self.learn_extra_mem = learn_extra_mem
        self.bias = bias

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        n_batch = tf.shape(x)[0]

        if self.n_extra_keys > 0:
            extra_keys = tf.get_variable("learned-keys",
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32, shape=(self.n_extra_keys, keys.shape.as_list()[-1]))
            keys = tf.concat([tf.tile(tf.expand_dims(extra_keys, 0), [n_batch, 1, 1]), keys], axis=1)
            if self.learn_extra_mem:
                extra_mems = tf.get_variable("learned-mems", shape=(self.n_extra_keys, memories.shape.as_list()[-1]),
                                             initializer=tf.zeros_initializer(), dtype=tf.float32)
            else:
                extra_mems = tf.constant(np.zeros((self.n_extra_keys, memories.shape.as_list()[-1]), dtype=np.float32))
            memories = tf.concat([tf.tile(tf.expand_dims(extra_mems, 0), [n_batch, 1, 1]), memories], axis=1)

        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        # joint_mask = compute_attention_mask(x_mask+self.n_extra_keys, mem_mask+self.n_extra_keys,
        #                                     x_word_dim+self.n_extra_keys, key_word_dim+self.n_extra_keys)
        joint_mask = compute_attention_mask(x_mask, mem_mask+self.n_extra_keys,
                                            x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.bias:
            # TODO its not clear if it better to do this or to just compute the softmax ourselves...
            atten_bas = tf.get_variable("atten-bias", (1, 1, 1), tf.float32,
                                        initializer=tf.zeros_initializer())
            atten_bas = tf.tile(atten_bas, [n_batch, tf.shape(x)[1], 1])
            dist_matrix = tf.concat([atten_bas, dist_matrix], axis=2)
            select_probs = tf.nn.softmax(dist_matrix)[:, :, 1:]
        else:
            select_probs = tf.nn.softmax(dist_matrix)

        #  Too (batch, x_word, memory_dim)
        response = tf.matmul(select_probs, memories)

        if self.merge is not None:
            with tf.variable_scope("merge"):
                response = self.merge.apply(response, x)
            return response
        else:
            return response


class StaticAttention(AttentionMapper):

    def __init__(self, attention: SimilarityFunction, merge: Optional[MergeLayer]=None, alignment_bias=None):
        self.attention = attention
        self.merge = merge
        self.alignment_bias = alignment_bias

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.alignment_bias is None:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        #  Too (batch, x_word, memory_dim)
        response = tf.matmul(select_probs, memories)

        if self.merge is not None:
            with tf.variable_scope("merge"):
                response = self.merge.apply(is_train, response, x)
            return response
        else:
            return response

    def __setstate__(self, state):
        if "alignment_bias" not in state["state"]:
            state["state"]["alignment_bias"] = None
        super().__setstate__(state)


class StaticAttentionWithEncoder(AttentionMapper):
    """ BiDaF like layer, except will allow the query vector to come from an arbitrary encoder layer """

    def __init__(self, attention: SimilarityFunction,
                 encoder_layer: SequenceEncoder,
                 alignment_bias=None):
        self.attention = attention
        self.encoder_layer = encoder_layer
        self.alignment_bias = alignment_bias

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.alignment_bias is None:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        #  Too (batch, x_word, memory_dim)
        response = tf.matmul(select_probs, memories)

        with tf.variable_scope("encode_keys"):
            encoded = self.encoder_layer.apply(is_train, keys, mem_mask)

        return tf.concat([x, response, x * response, x * tf.expand_dims(encoded, 1)], axis=2)


class StaticAttentionLearnedMemories(AttentionMapper):

    def __init__(self, attention: SimilarityFunction,
                 encoder_layer: SequenceEncoder,
                 n_learned: int,
                 same_key_and_memory: bool,
                 bi_attention: bool):
        self.bi_attention = bi_attention
        self.attention = attention
        self.n_learned = n_learned
        self.encoder_layer = encoder_layer
        self.same_key_and_memory = same_key_and_memory

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]
        batch_dim = tf.shape(keys)[0]

        if self.n_learned > 0:
            if not self.same_key_and_memory:
                raise NotImplementedError()
            key_size = keys.shape.as_list()[-1]
            learned_keys = tf.get_variable("learned_key", (self.n_learned, key_size))
            tiled = tf.tile(tf.expand_dims(learned_keys, 0), [batch_dim, 1, 1])
            keys = tf.concat([tiled, keys], axis=1)
            memories = tf.concat([tiled, memories], axis=1)
            mem_mask += self.n_learned

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, keys)

        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        select_probs = tf.nn.softmax(dist_matrix)

        response = tf.matmul(select_probs, memories)

        out = [x, response, x * response]

        if self.bi_attention:
            context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
            context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)

            # batch mult (1, x_words) matrice by (x_words, x_dim) to get (1, x_dim)
            select_context = tf.matmul(tf.expand_dims(context_probs, 1), memories)  # (batch, 1, x_dim)
            out.append(x * tf.expand_dims(select_context, 1))

        return tf.concat(out, axis=2)


class StaticAttentionSelf(SequenceMapper):

    def __init__(self, attention: SimilarityFunction,
                 merge: Optional[MergeLayer]=None,
                 alignment_bias=False):
        self.alignment_bias = alignment_bias
        self.attention = attention
        self.merge = merge

    def apply(self, is_train, x, x_mask=None):
        x_word_dim = tf.shape(x)[1]

        # (batch, x_word, key_word)
        dist_matrix = self.attention.get_scores(x, x)
        dist_matrix += tf.expand_dims(tf.eye(x_word_dim) * VERY_NEGATIVE_NUMBER, 0)

        joint_mask = compute_attention_mask(x_mask, x_mask, x_word_dim, x_word_dim)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))

        if self.alignment_bias is None:
            select_probs = tf.nn.softmax(dist_matrix)
        else:
            bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
            dist_matrix = tf.exp(dist_matrix)
            select_probs = dist_matrix / (tf.reduce_sum(dist_matrix, axis=2, keep_dims=True) + bias)

        response = tf.matmul(select_probs, x)  # (batch, x_words, q_dim)

        if self.merge is not None:
            with tf.variable_scope("merge"):
                response = self.merge.apply(is_train, response, x)
            return response
        else:
            return response

    def __setstate__(self, state):
        if "alignment_bias" not in state["state"]:
            state["state"]["alignment_bias"] = False
        super().__setstate__(state)


class BiAttention(AttentionMapper):
    """ Bi-attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, sim: SimilarityFunction, q2c: bool, query_dots: bool=True):
        self.sim = sim
        self.q2c = q2c
        self.query_dots = query_dots

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None):
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        dist_matrix = self.sim.get_scores(x, keys)
        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
        query_probs = tf.nn.softmax(dist_matrix)  # probability of each mem_word per x_word

        # Batch matrix multiplication to get the attended vectors
        select_query = tf.matmul(query_probs, memories)  # (batch, x_words, q_dim)

        if not self.q2c:
            if self.query_dots:
                return tf.concat([x, select_query, x * select_query], axis=2)
            else:
                return tf.concat([x, select_query], axis=2)

        # select query-to-context
        context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
        context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)
        select_context = tf.einsum("ai,aik->ak", context_probs, x)  # (batch, x_dim)
        select_context = tf.expand_dims(select_context, 1)

        if self.query_dots:
            return tf.concat([x, select_query, x * select_query, x * select_context], axis=2)
        else:
            return tf.concat([x, select_query, x * select_context], axis=2)

    def __setstate__(self, state):
        if "query_dots" not in state["state"]:
            state["state"]["query_dots"] = True
        super().__setstate__(state)


class AttentionEncoder(SequenceEncoder):
    def __init__(self, key_mapper: SequenceMapper=None,
                 post_process: Mapper=None,
                 init="glorot_uniform"):
        self.init = init
        self.key_mapper = key_mapper
        self.post_process = post_process

    def apply(self, is_train, x, mask=None):
        if self.key_mapper is not None:
            with tf.variable_scope("map_keys"):
                keys = self.key_mapper.apply(is_train, x, mask)
        else:
            keys = x

        weights = tf.get_variable("weights", keys.shape.as_list()[-1], dtype=tf.float32,
                                  initializer=get_keras_initialization(self.init))
        dist = tf.tensordot(keys, weights, axes=[[2], [0]])  # (batch, x_words)
        dist = exp_mask(dist, mask)
        dist = tf.nn.softmax(dist)

        out = tf.einsum("ajk,aj->ak", x, dist)  # (batch, x_dim)

        if self.post_process is not None:
            with tf.variable_scope("post_process"):
                out = self.post_process.apply(is_train, out)
        return out


class MultiAttentionEncoder(SequenceMultiEncoder):
    def __init__(self, n_encodings: int, bias: bool=False, key_mapper: SequenceMapper=None,
                 post_process: Mapper=None,
                 init="glorot_uniform"):
        self.init = init
        self.bias = bias
        self.n_encodings = n_encodings
        self.key_mapper = key_mapper
        self.post_process = post_process

    def apply(self, is_train, x, mask=None):
        if self.key_mapper is not None:
            with tf.variable_scope("map_keys"):
                keys = self.key_mapper.apply(is_train, x, mask)
        else:
            keys = x

        weights = tf.get_variable("weights", (keys.shape.as_list()[-1], self.n_encodings), dtype=tf.float32,
                                  initializer=get_keras_initialization(self.init))
        dist = tf.tensordot(keys, weights, axes=[[2], [0]])  # (batch, x_words, n_encoding)
        if self.bias:
            dist += tf.get_variable("bias", (1, 1, self.n_encodings),
                                    dtype=tf.float32, initializer=tf.zeros_initializer())
        if mask is not None:
            bool_mask = tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)
            dist = bool_mask * bool_mask + (1 - bool_mask) * VERY_NEGATIVE_NUMBER

        dist = tf.nn.softmax(dist, dim=1)

        out = tf.einsum("ajk,ajn->ank", x, dist)  # (batch, n_encoding, feature)

        if self.post_process is not None:
            with tf.variable_scope("post_process"):
                out = self.post_process.apply(is_train, out)
        return out


class MultiSelfAttention(SequenceMapper):
    def __init__(self, n_heads: int, project_size: Optional[int], memory_size: Optional[int]=None,
                 shared_project: bool=False, project_bias: bool=False, bilinear_comp: bool=False,
                 init= "glorot_uniform", merge: Optional[MergeLayer]=None, scale=True, bias=True):
        self.n_heads = n_heads
        self.bilinear_comp = bilinear_comp
        self.merge = merge
        self.project_bias = project_bias
        self.project_size = project_size
        self.shared_project = shared_project
        self.memory_size = memory_size
        self.scale = scale
        self.bias = bias
        self.init = init

    def apply(self, is_train, x, mask=None):
        batch_size = tf.shape(x)[0]
        x_word_dim = tf.shape(x)[1]
        x_feature_dim = x.shape.as_list()[-1]
        project_size = self.project_size
        if project_size is None:
            project_size = x_feature_dim // self.n_heads
            if x_feature_dim % self.n_heads != 0:
                raise ValueError()
        mem_size = self.memory_size
        if mem_size is None:
            mem_size = project_size

        init = get_keras_initialization(self.init)

        query_proj = tf.get_variable("query_proj", (x_feature_dim, self.n_heads, project_size), initializer=init)
        if self.shared_project:
            key_proj = query_proj
        else:
            key_proj = tf.get_variable("key_proj", (x_feature_dim, self.n_heads, project_size), initializer=init)
        mem_proj = tf.get_variable("mem_proj", (x_feature_dim, self.n_heads, mem_size), initializer=init)

        queries = tf.tensordot(x, query_proj, [[2], [0]])  # (batch, word, n_head, project_size)
        keys = tf.tensordot(x, key_proj, [[2], [0]])  # (batch, key, n_head, project_size)

        if self.project_bias:
            queries += tf.get_variable("query_bias", (1, 1, self.n_heads, project_size),
                                        initializer=tf.zeros_initializer())
            keys += tf.get_variable("key_bias", (1, 1, self.n_heads, project_size),
                                        initializer=tf.zeros_initializer())

        # dist_matrix = tf.matmul(queries, keys, transpose_b=True)
        dist_matrix = tf.einsum("bwhd,bkhd->bwkh", queries, keys)   # dots of (batch, word, key, head)

        if self.scale:
            dist_matrix /= tf.sqrt(float(project_size))

        if self.bilinear_comp:
            query_bias_proj = tf.get_variable("query_bias_proj", (x_feature_dim, self.n_heads), initializer=init)
            key_bias_proj = tf.get_variable("query_bias_proj", (x_feature_dim, self.n_heads), initializer=init)
            dist_matrix += tf.expand_dims(tf.tensordot(x, query_bias_proj, [[2], [0]]), 2)
            dist_matrix += tf.expand_dims(tf.tensordot(x, key_bias_proj, [[2], [0]]), 1)

        joint_mask = compute_attention_mask(mask, mask, x_word_dim, x_word_dim)
        if joint_mask is not None:
            dist_matrix += tf.expand_dims(VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype)), 2)
        dist_matrix += tf.expand_dims(tf.expand_dims(tf.eye(x_word_dim) * VERY_NEGATIVE_NUMBER, 0), 2)

        if self.bias:
            bias = tf.get_variable("bias", (1, 1, self.n_heads, 1), initializer=tf.zeros_initializer())
            dist_matrix += bias

        select_probs = tf.nn.softmax(dist_matrix)   # for each (batch, word, head) probability over keys

        memories = tf.tensordot(x, mem_proj, [[2], [0]])  # (batch, memory, head, mem_size)
        response = tf.einsum("bwhk,bkhd->bwhd", select_probs, memories)  # (batch, word, head, mem_size)

        response = tf.reshape(response, (batch_size, x_word_dim, self.n_heads * mem_size))   # concat heads

        if self.merge is not None:
            with tf.variable_scope("merge"):
                 response = self.merge.apply(is_train, x, response)
        return response
