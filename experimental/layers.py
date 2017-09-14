

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


class LearnedDropoutIndependentLayer(Updater):
    def __init__(self, keep_probs, layer: Optional[Updater] = None, metric="l2", n_drop=None):
        self.layer = layer
        self.n_drop = n_drop
        self.keep_probs = keep_probs
        self.metric = metric

    @property
    def version(self):
        # v2: correctly stop gradients from norm to `drop_x`
        # v3: only penalize the correction to the drop-in entries
        return 3

    def apply(self, is_train, x, mask=None):
        if self.keep_probs >= 1:
            return x

        random_tensor = self.keep_probs + tf.random_uniform(tf.shape(x), minval=0, maxval=1, dtype=x.dtype)
        if self.n_drop is not None:
            n_keep = x.shape.as_list()[-1] - self.n_drop
            to_add = tf.concat([tf.zeros([1, 1, self.n_drop]), tf.ones([1, 1, n_keep])], axis=2)
            random_tensor += to_add

        binary_tensor = tf.floor(random_tensor)  # 1 if keep, 0 if dropped
        entries_dropped = (1 - binary_tensor)  # 0 if keep, 1 if dropped
        drop_x = x * binary_tensor

        if self.layer is None:
            fill_in = tf.get_variable("dropout_fill_values", x.shape.as_list()[-1],
                                      initializer=tf.zeros_initializer())
            for i in range(len(x.shape) - 1):
                fill_in = tf.expand_dims(fill_in, 0)
        else:
            fill_in = self.layer.apply(is_train, tf.stop_gradient(drop_x), mask)
            if self.n_drop is not None:
                keep_shape = tf.shape(x)
                keep_shape = [keep_shape[0], keep_shape[1], keep_shape[2] - self.n_drop]
                fill_in = tf.concat([fill_in, tf.zeros(keep_shape)], axis=2)

        if mask is not None:
            # the masked entries are counted as 'keep'
            entries_dropped *= tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)

        if self.metric == "l2" or self.metric is None:
            reg = tf.reduce_mean(tf.pow((tf.stop_gradient(x) - fill_in)*entries_dropped, 2))
        elif self.metric == "l1":
            reg = tf.reduce_mean(tf.abs((tf.stop_gradient(x) - fill_in)*entries_dropped))
        else:
            raise ValueError()

        tf.add_to_collection("auxillary_losses", reg)

        return tf.cond(is_train, lambda: drop_x + tf.stop_gradient(fill_in) * entries_dropped, lambda: x)


class LearnedDropoutLayer(Updater):
    def __init__(self, keep_probs, layer: Optional[Updater]=None):
        self.layer = layer
        self.keep_probs = keep_probs

    def apply(self, is_train, x, mask=None):
        if self.keep_probs >= 1:
            return x
        return tf.cond(is_train, lambda : self._dropout(x, is_train, mask), lambda : x)

    def _dropout(self, x, is_train, mask):
        random_tensor = self.keep_probs + tf.random_uniform(tf.shape(x), minval=0, maxval=1, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        drop_x = x * binary_tensor

        if self.layer is None:
            fill_in = tf.get_variable("dropout_fill_values", x.shape.as_list()[-1],
                                       initializer=tf.zeros_initializer())
            for i in range(len(x.shape)-1):
                fill_in = tf.expand_dims(fill_in, 0)
        else:
            fill_in = self.layer.apply(is_train, drop_x, mask)

        if mask is not None:
            fill_in *= tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)

        return drop_x + fill_in * (1 - binary_tensor)




class FillInLayer(SequenceMapper):
    def __init__(self, mapper: SequenceMapper):
        self.mapper = mapper

    def apply(self, is_train, x, mask=None):
        out = self.mapper.apply(is_train, x, mask)
        return x + (out * (x == 0))



class BiDropoutLayer(Updater):
    def __init__(self, drop_by: float=0, percent_lower: float=0.5):
        self.drop_by = drop_by
        self.percent_lower = percent_lower

    def apply(self, is_train, x, mask=None):
        return tf.cond(is_train, lambda: self._drop(x), lambda: x)

    def _drop(self, x):
        split = tf.cast(tf.random_uniform(tf.shape(x), 0, 1) < self.percent_lower, tf.float32)
        return (x * split) * self.drop_by + \
               (x*(1-split)) * (1 - self.drop_by * self.percent_lower) / (1 - self.percent_lower)


class NormalDropoutLayer(Updater):
    def __init__(self, var: float=0):
        self.var = var

    def apply(self, is_train, x, mask=None):
        return tf.cond(is_train, lambda: x * tf.random_normal(tf.shape(x), 1, tf.sqrt(self.var)), lambda: x)


class SoftDropoutLayer(Updater):
    def __init__(self, keep_probs: float):
        self.keep_prob = keep_probs

    def apply(self, is_train, x, mask=None):
        return tf.cond(is_train, lambda: x * tf.random_uniform(tf.shape(x), 1.0-self.keep_prob, 1.0+self.keep_prob), lambda: x)


class PartialDropoutLayer(Updater):
    def __init__(self, keep_probs: float, n_keep):
        self.keep_probs = keep_probs
        self.n_keep = n_keep

    def apply(self, is_train, x, mask=None):
        last_dim = x.shape.as_list()[-1]
        weights = tf.concat([
            tf.constant(self.keep_probs, tf.float32, (last_dim-self.n_keep,)),
            tf.constant(1, tf.float32, (self.n_keep,))
        ], axis=0)
        return mixed_dropout(is_train, x, weights)


class BottleneckDropoutLayer(Updater):
    def __init__(self, keep_probs: float, n_units: int, share_weights: bool=False, init="glorot_uniform"):
        self.keep_probs = keep_probs
        self.init = init
        self.n_units = n_units
        self.share_weights = share_weights

    def apply(self, is_train, x, mask=None):
        rank = len(x.shape) - 1
        init = get_keras_initialization(self.init)
        input_dim = x.shape.as_list()[-1]
        encode_w = tf.get_variable("encode_weights", shape=(input_dim, self.n_units),
                                   dtype=tf.float32, initializer=init)
        if self.share_weights:
            decode_w = tf.transpose(encode_w)
        else:
            decode_w = tf.get_variable("decode_weights", shape=(self.n_units, input_dim),
                                       dtype=tf.float32, initializer=init)
        encoded = tf.tensordot(x, encode_w, [[rank], [0]])
        decoded = tf.tensordot(encoded, decode_w, [[rank], [0]])
        return dropout(x, self.keep_probs, is_train) + decoded


class BottleneckDropoutConcatLayer(Updater):
    def __init__(self, keep_probs: float, n_units: int, init="glorot_uniform"):
        self.keep_prob = keep_probs
        self.init = init
        self.n_units = n_units

    def apply(self, is_train, x, mask=None):
        rank = len(x.shape) - 1
        init = get_keras_initialization(self.init)
        input_dim = x.shape.as_list()[-1]
        encode_w = tf.get_variable("encode_weights", shape=(input_dim, self.n_units),
                                   dtype=tf.float32, initializer=init)
        encoded = tf.tensordot(x, encode_w, [[rank], [0]])
        return tf.concat([dropout(x, self.keep_prob, is_train), encoded], axis=rank)



class MultiAggregateLayer(SequenceEncoder):
    def __init__(self, reduce: List[str], apply_mask=True):
        self.reduce = reduce
        self.apply_mask = apply_mask

    def apply(self, is_train, x, mask=None):
        if mask is not None:
            answer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(x)[1]), tf.float32), 2)
            if self.apply_mask:
                x *= answer_mask
        else:
            answer_mask = None

        out = []
        for r in self.reduce:
            if r == "max":
                if mask is not None:
                    out.append(tf.reduce_max(x+((tf.reduce_min(x)-1) * (1 - answer_mask)), axis=1))
                else:
                    out.append(tf.reduce_max(x, axis=1))
            elif r == "sum":
                return tf.reduce_sum(x, axis=1)
            elif r == "mean":
                if mask is not None:
                    out.append((tf.reduce_sum(x, axis=1) / tf.expand_dims(mask, 1)))
                else:
                    out.append(tf.reduce_mean(x, axis=1))
            else:
                raise ValueError()
        return tf.concat(out, axis=1)





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



#
# class CudnnGRULayer(SequenceMapper):
#     def __init__(self, n_units, n_layers, direction, weight_init=None, bias_init=None):
#         self.n_units = n_units
#         self.n_layers = n_layers
#         self.direction = direction
#         self.weight_init = weight_init
#         self.bias_init = bias_init
#
#     def apply(self, is_train, x, mask=None):
#         rnn = CudnnGRU(self.n_units, self.n_layers, self.direction)
#         rnn.
#         w = tf.get_variable("weights", )
#         rnn.canonical_to_params()
#
#
#         self._rnn.canonical_to_params()
