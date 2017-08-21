import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

from nn.recurrent_layers import CudnnLstm, BiDirectionalFusedLstm


def lstm():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    b = tf.placeholder(tf.bool, ())
    x = tf.placeholder(tf.float32, (10, 10, 10))
    seq_len = tf.placeholder(tf.int32, 10)
    tmp = CudnnLstm(100, 1)
    with sess.as_default():
        out = tmp.apply(b, x, seq_len)
    for v in tf.trainable_variables():
        print(v)


def fused_lstm():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    b = tf.placeholder(tf.bool, ())
    x = tf.placeholder(tf.float32, (10, 10, 10))
    seq_len = tf.placeholder(tf.int32, 10)
    tmp = BiDirectionalFusedLstm(100)
    with sess.as_default():
        out = tmp.apply(b, x, seq_len)
    for v in tf.trainable_variables():
        print(v)


def test_recurrent():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    input_data = tf.zeros((1, 1, 10))
    input_data_shape = input_data.get_shape()

    feature_dim = input_data_shape[2].value

    n_units = 20
    cell = cudnn_rnn_ops.CudnnGRU(
      1, n_units, feature_dim, direction="unidirectional", dropout=0,
        input_mode="linear_input")

    w, b = cell.params_to_canonical(tf.zeros([cell.params_size()]))
    is_recurrent = [False, False, False, True, True, True]

    with sess.as_default():
        n_params = cell.params_size().eval()
        weights, _ = cell.params_to_canonical(tf.zeros([n_params]))
        biases = [tf.zeros(n_units) for _ in range(len(b))]

        init_weights = []
        print([tf.shape(w).eval() for w in weights])
        for w, r in zip(weights, is_recurrent):
            if r:
                # init_weights.append(tf.reshape(tf.eye(n_units, dtype=w.dtype), tf.shape(w)))
                init_weights.append(tf.reshape(tf.eye(n_units, dtype=w.dtype), tf.shape(w)))
            else:
                init_weights.append(tf.zeros(tf.shape(w).eval(), w.dtype))
        out = cell.canonical_to_params(init_weights, biases)
        out.set_shape((n_params,))

    x = tf.zeros_like(input_data)
    # x = tf.random_uniform(tf.shape(input_data), maxval=1)
    # out = tf.random_uniform((n_params,), -1, 1)
    print("HERE!")
    step = cell.__call__(x, tf.ones((1, 1, n_units)), out)

    out = sess.run(step)
    for x in out:
        print(x)


def test_bais():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    input_data = tf.zeros((1, 1, 300))
    input_data_shape = input_data.get_shape()

    time_dim = input_data_shape[1].value
    batch_dim = input_data_shape[1].value
    feature_dim = input_data_shape[2].value

    n_units = 80
    cell = cudnn_rnn_ops.CudnnLSTM(
      1, n_units, feature_dim, direction="unidirectional", dropout=0,
        input_mode="linear_input")

    w, b = cell.params_to_canonical(tf.zeros([cell.params_size()]))
    print(len(b))

    with sess.as_default():
        n_params = cell.params_size().eval()
        weights, _ = cell.params_to_canonical(tf.zeros([n_params]))
        weights = [tf.zeros(tf.shape(w).eval(), w.dtype) for w in weights]
        biases = [tf.constant(0.5, tf.float32, (n_units,)) if z else tf.zeros(n_units) for z in
                  [False, True, False, False, False, True, False, False]]
        out = cell.canonical_to_params(weights, biases)
        out.set_shape((n_params,))

    x = tf.zeros_like(input_data)
    step = cell.__call__(x, tf.zeros((1, time_dim, n_units)), tf.ones((1, time_dim, n_units)), out)[1:]

    out = sess.run(step)
    for x in out:
        print(x)
    # # # sess.run(cell(input_data, input_h, params))
    # #
    # params_saveable = cudnn_rnn_ops.RNNParamsSaveable(
    #     cell.params_to_canonical,
    #     cell.canonical_to_params,
    #     "%s/params_canonical" % "", params)
    # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    # # print(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
    # #
    # saver = tf.train.Saver(max_to_keep=2, var_list=[params_saveable])
    #
    # with tf.device("/gpu:0"):
    #     print("Saving...")
    #     saver.save(sess, "tmp/tmp")
    #     print("Done!")

if __name__ == "__main__":
    test_recurrent()