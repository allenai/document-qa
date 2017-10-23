import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from docqa.nn.recurrent_layers import _compute_gates


class TestInitLstm(unittest.TestCase):

    def test_forget_bias(self):
        """
        Make sure the forget bias is only being applied to the forget gate
        """
        batches = 1
        num_units = 5
        num_inputs = 5

        hidden_size = (batches, num_units)
        input_size = (batches, num_inputs)

        inputs = tf.placeholder(dtype='float32', shape=input_size)
        h = tf.placeholder(dtype='float32', shape=hidden_size)
        with tf.variable_scope("test_bias"):
            i_t, j_t, f_t, o_t = _compute_gates(inputs, h, 4 * num_units, 1,
                                                init_ops.zeros_initializer(), init_ops.zeros_initializer())
        gates = [i_t, j_t, f_t, o_t]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Make sure the bias is ONLY getting applied to the forget gate
        [i,j,f,o] = sess.run(gates, feed_dict={inputs: np.zeros(input_size), h: np.ones(hidden_size)})
        self.assertTrue(np.allclose(f, np.ones(f.shape), rtol=0))
        for x in [i,j,o]:
            self.assertTrue(np.allclose(x, np.zeros(x.shape), rtol=0))

    def test_inits(self):
        """
        Make sure the initializers effects the correct weights
        """
        batches = 1
        num_units = 2
        num_inputs = 3

        hidden_size = (batches, num_units)
        input_size = (batches, num_inputs)

        inputs = tf.placeholder(dtype='float32', shape=input_size)
        h = tf.placeholder(dtype='float32', shape=hidden_size)
        with tf.variable_scope("test_inits"):
            i_t, j_t, f_t, o_t = _compute_gates(inputs, h, num_units, 0,
                                                init_ops.constant_initializer(1), init_ops.constant_initializer(100))
        gates = [i_t, j_t, f_t, o_t]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        inputs_init = np.zeros(input_size)
        hidden_init = np.zeros(hidden_size)
        inputs_init[0] = 1
        i, j, f, o = sess.run(gates, feed_dict={inputs: inputs_init, h: hidden_init})
        self.assertTrue(np.allclose(i, np.full(i.shape, num_inputs), rtol=0))

        inputs_init[0] = 0
        hidden_init[0] = 1
        i, j, f, o = sess.run(gates, feed_dict={inputs: inputs_init, h: hidden_init})
        self.assertTrue(np.allclose(i, np.full(i.shape, num_units*100), rtol=0))

        hidden_init[0] = 0
        inputs_init[0, 0] = -2
        hidden_init[0, 0] = 1
        i, j, f, o = sess.run(gates, feed_dict={inputs: inputs_init, h: hidden_init})
        self.assertTrue(np.allclose(i, np.full(i.shape, 98), rtol=0))
