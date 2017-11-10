
import tensorflow as tf

from docqa.nn.layers import Mapper


class ElmoLayer(Mapper):
    def __init__(self, l2_coef: float, layer_norm: bool, top_layer_only: bool):
        self.l2_coef = l2_coef
        self.layer_norm = layer_norm
        self.top_layer_only = top_layer_only

    def apply(self, is_train, x, mask=None):
        mask = tf.sequence_mask(mask, tf.shape(x)[1])
        output = weight_layers(1, x, mask, self.l2_coef, do_layer_norm=self.layer_norm,
                               use_top_only=self.top_layer_only)["weighted_ops"][0]
        return output

    def __setstate__(self, state):
        if "softmax" not in state:
            state["softmax"] = True
        if "layer_norm" not in state:
            state["layer_norm"] = True
        if "top_layer_only" not in state:
            state["top_layer_only"] = False
        super().__setstate__(state)


def weight_layers(n_out_layers, lm_embeddings, mask, l2_coef=None,
                  use_top_only=False, do_layer_norm=True):
    '''
    Weight the layers of a biLM with trainable scalar weights.
    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES
    Input:
        n_out_layers: the number of weighted output layers
        bidirectional_lm: an instance of BidirectionalLanguageModel
        l2_coef: the l2 regularization coefficient
    Output:
        {
            'weighted_ops': [
                op to compute weighted average for output layer1,
                op to compute regularization term for output layer2,
                ...
            ],
            'regularization_ops': [
                op to compute regularization term for output layer1,
                op to compute regularization term for output layer2,
                ...
            ]
        }
    '''

    def _l2_regularizer(weights):
        return l2_coef * tf.reduce_sum(tf.square(weights))

    # Get ops for computing LM embeddings and mask
    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])

    if not tf.get_variable_scope().reuse:
        prefix = "monitor/"
        if "weight_embed" in tf.get_variable_scope().name:
            prefix += "input/"
        else:
            prefix += "output/"
    else:
        prefix = None

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                     ) / N
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )

        ret = {'weighted_ops': [], 'regularization_ops': []}
        for k in range(n_out_layers):
            if use_top_only:
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
                # just the top layer
                sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
                # no regularization
                reg = [0.0]
            else:
                W = tf.get_variable(
                    'ELMo_W_{}'.format(k),
                    shape=(n_lm_layers,),
                    initializer=tf.zeros_initializer,
                    regularizer=_l2_regularizer,
                    trainable=True,
                )

                if prefix is not None:
                    for i in range(3):
                        print("Monitoring " + prefix + "%d/" % i)
                        tf.add_to_collection(prefix + "%d/" % i, W[i])

                # normalize the weights
                normed_weights = tf.split(
                    tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
                )
                # split LM layers
                layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

                # compute the weighted, normalized LM activations
                pieces = []
                for w, t in zip(normed_weights, layers):
                    if do_layer_norm:
                        pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                    else:
                        pieces.append(w * tf.squeeze(t, squeeze_dims=1))
                sum_pieces = tf.add_n(pieces)

                # get the regularizer
                reg = [
                    r for r in tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES)
                    if r.name.find('ELMo_W_{}/'.format(k)) >= 0
                ]

            # scale the weighted sum by gamma
            gamma = tf.get_variable(
                'ELMo_gamma_{}'.format(k),
                shape=(1,),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
            )

            if prefix is not None:
                tf.add_to_collection(prefix + "gamma", gamma[0])

            weighted_lm_layers = sum_pieces * gamma

            ret['weighted_ops'].append(weighted_lm_layers)
            ret['regularization_ops'].append(reg[0])

    return ret