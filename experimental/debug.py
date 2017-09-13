mport tensorflow as tf
class DebugPredictor(SequencePredictionLayer):
    def apply(self, is_train, context_embed, answer, context_mask=None):

        context_embed = tf.Print(context_embed, [tf.reduce_all(tf.is_finite(context_embed))], "c")

        with tf.variable_scope("start_pred"):
            logits1 = fully_connected(context_embed, 1, activation_fn=None)
            logits1 = tf.squeeze(logits1, squeeze_dims=[2])

        with tf.variable_scope("end_pred"):
            logits2 = fully_connected(context_embed, 1, activation_fn=None)
            logits2 = tf.squeeze(logits2, squeeze_dims=[2])

        masked_start_logits = logits1
        masked_end_logits = logits2

        # if len(answer) == 1:
        #     # answer span is encoding in a sparse int array
        #     answer_spans = answer[0]
        #     losses1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=masked_start_logits, labels=answer_spans[:, 0])
        #     losses2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=masked_end_logits, labels=answer_spans[:, 1])
        #     loss = tf.add_n([tf.reduce_mean(losses1), tf.reduce_mean(losses2)], name="loss")
        #
        if len(answer) == 2 and all(x.dtype == tf.bool for x in answer):
            # all correct start/end bounds are marked in a dense bool array
            # In this case there might be multiple answer spans, so we need an aggregation strategy
            losses = []
            for answer_mask, logits in zip(answer, [masked_start_logits, masked_end_logits]):
                answer_mask = tf.cast(answer_mask, tf.float32)
                log_norm = tf.reduce_logsumexp(logits, axis=1)
                log_score = tf.reduce_logsumexp(logits * answer_mask, axis=1)
                losses.append(tf.reduce_mean(-(log_score - log_norm)))
            loss = tf.add_n(losses)
            loss = tf.Print(loss, [loss], "loss")
        else:
            raise RuntimeError()

        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return BoundaryPrediction(tf.nn.softmax(masked_start_logits),
                                  tf.nn.softmax(masked_end_logits),
                                  masked_start_logits, masked_end_logits)

class DebugCudnnGru():
    def __init__(self,
                 n_units,
                 n_layers=1,
                 keep_recurrent=1,
                 w_init="truncated_normal",
                 recurrent_init=None,
                 bidirectional=True,
                 learn_initial_states=False,
                 save_cannonical_parameters=True):
        if bidirectional is None or n_layers is None or n_units is None:
            raise ValueError()
        self._kind = "GRU"
        self.keep_recurrent = keep_recurrent
        self.n_units = n_units
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.w_init = w_init
        self.recurrent_init = recurrent_init
        self.learn_initial_states = learn_initial_states
        self.save_cannonical_parameters = save_cannonical_parameters

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


        parameters = tf.get_variable(
            "gru_parameters",
            n_params,
            tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.05)
        )

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

    def apply(self, is_train, x, mask=None):
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



def debug_model(char_th: int, dim: int, mode: str, preprocess):
    print('WARNING USE DEBUG MODEL')

    if mode == "paragraph" or mode == "merge":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = DebugPredictor()
    else:
        raise NotImplementedError(mode)


    # return ContextOnly(
    #     encoder=DocumentAndQuestionEncoder(answer_encoder),
    #     word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False, cpu=True),
    #     char_embed=None,
    #     context_encoder=recurrent_layer,
    #     prediction=BoundsPredictor(NullBiMapper())
    # )

    return Attention(
        encoder=DocumentAndQuestionEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=None,
        # char_embed=CharWordEmbedder(
        #     LearnedCharEmbedder(word_size_th=14, char_th=char_th, char_dim=20, init_scale=0.05, force_cpu=True),
        #     MaxPool(Conv1d(100, 5, 0.8)),
        #     shared_parameters=True
        # ),
        preprocess=preprocess,
        word_embed_layer=None,
        embed_mapper=None,
        question_mapper=None,
        context_mapper=NullMapper(),
        memory_builder=NullBiMapper(),
        attention=NullAttention(),
        match_encoder=DebugCudnnGru(dim, w_init=TruncatedNormal()),
        predictor=predictor
    )

