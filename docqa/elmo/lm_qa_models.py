from os.path import join, expanduser
from typing import Optional, List, Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from docqa.configurable import Configurable
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextAndQuestion
from docqa.elmo.data import Batcher, TokenBatcher
from docqa.elmo.lm_model import BidirectionalLanguageModel
from docqa.encoder import DocumentAndQuestionEncoder
from docqa.model import Model, Prediction
from docqa.nn.embedder import WordEmbedder, CharWordEmbedder
from docqa.nn.layers import SequenceMapper, SequencePredictionLayer, SequenceBiMapper, AttentionMapper, \
    AttentionPredictionLayer, Mapper
from docqa.utils import ResourceLoader, flatten_iterable

LM_DIR = join(expanduser("~"), "data", "lm")


class LanguageModel(Configurable):
    """ Pointer to the source files needed to use a pre-trained language model """
    def __init__(self, lm_vocab_file, options_file, weight_file, embed_weights_file: Optional[str]):
        self.lm_vocab_file = lm_vocab_file
        self.options_file = options_file
        self.weight_file = weight_file
        self.embed_weights_file = embed_weights_file


class SquadContextConcatSkip(LanguageModel):
    """ Model re-trained on SQuAD that uses skip connections"""
    def __init__(self):
        basedir = join(LM_DIR, "squad-context-concat-skip")
        super().__init__(
            join(basedir, "squad_train_dev_all_unique_tokens.txt"),
            join(basedir, "options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json"),
            join(basedir, "squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5"),
            join(basedir, "squad_train_dev_all_unique_tokens_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5"),
        )


class ElmoQaModel(Model):
    """ Base classes for ELMo models """
    def __init__(self,
                 encoder: DocumentAndQuestionEncoder,
                 lm_model: LanguageModel,
                 per_sentence: bool,
                 max_batch_size: int,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder] = None):
        if word_embed is None and char_embed is None:
            raise ValueError()

        self.max_batch_size = max_batch_size
        self.lm_model = lm_model
        self.per_sentence = per_sentence
        self.preprocessor = None
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.encoder = encoder
        if self.per_sentence:
            self._max_num_sentences = self.max_batch_size * 30  # TODO hard coded for SQuAD
        else:
            self._max_num_sentences = self.max_batch_size
        self._batcher = None
        self._max_word_size = None

        # placeholders
        self._is_train_placeholder = None
        self._batch_len_placeholders = None
        self._question_char_ids_placeholder = None
        self._context_char_ids_placeholder = None
        self._context_sentence_ixs = None

    @property
    def token_lookup(self):
        """
        Are we using pre-computed word vectors, or running the LM's CNN to dynmacially derive
        word vectors from characters.
        """
        return self.lm_model.embed_weights_file is not None

    def init(self, corpus, loader: ResourceLoader):
        if self.word_embed is not None:
            self.word_embed.set_vocab(corpus, loader,
                                      None if self.preprocessor is None else self.preprocessor.special_tokens())
        if self.char_embed is not None:
            self.char_embed.embeder.set_vocab(corpus)

    def set_inputs(self, datasets: List[ParagraphAndQuestionDataset], word_vec_loader=None):
        voc = set()
        for dataset in datasets:
            voc.update(dataset.get_vocab())

        input_spec = datasets[0].get_spec()
        for dataset in datasets[1:]:
            input_spec += dataset.get_spec()

        return self.set_input_spec(input_spec, voc, word_vec_loader)

    def set_input_spec(self, input_spec, voc, word_vec_loader=None):
        if word_vec_loader is None:
            word_vec_loader = ResourceLoader()
        if self.word_embed is not None:
            self.word_embed.init(word_vec_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(word_vec_loader, voc)

        batch_size = input_spec.batch_size
        self.encoder.init(input_spec, True, self.word_embed,
                          None if self.char_embed is None else self.char_embed.embeder)
        self._is_train_placeholder = tf.placeholder(tf.bool, ())

        if self.token_lookup:
            self._batcher = TokenBatcher(self.lm_model.lm_vocab_file)
            self._question_char_ids_placeholder = tf.placeholder(tf.int32, (batch_size, None))
            self._context_char_ids_placeholder = tf.placeholder(tf.int32, (batch_size, None))
            self._max_word_size = input_spec.max_word_size
            self._context_sentence_ixs = None
        else:
            input_spec.max_word_size = 50  # TODO hack, harded coded from the lm model
            self._batcher = Batcher(self.lm_model.lm_vocab_file, 50)
            self._max_word_size = input_spec.max_word_size
            self._question_char_ids_placeholder = tf.placeholder(tf.int32,
                                                                 (batch_size, None, self._max_word_size))
            if self.per_sentence:
                self._context_char_ids_placeholder = tf.placeholder(tf.int32,
                                                                    (None, None, self._max_word_size))
                self._context_sentence_ixs = tf.placeholder(tf.int32, (batch_size, 3, None, 3))
            else:
                self._context_char_ids_placeholder = tf.placeholder(tf.int32,
                                                                    (batch_size, None, self._max_word_size))
                self._context_sentence_ixs = None

        return self.get_placeholders()

    def get_placeholders(self):
        return self.encoder.get_placeholders() + [
            self._is_train_placeholder,
            self._question_char_ids_placeholder,
            self._context_char_ids_placeholder
        ] + ([self._context_sentence_ixs] if (self._context_sentence_ixs is not None) else [])

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]):
        is_train = input_tensors[self._is_train_placeholder]
        enc = self.encoder

        q_lm_model = BidirectionalLanguageModel(self.lm_model.options_file, self.lm_model.weight_file,
                                                input_tensors[self._question_char_ids_placeholder],
                                                embedding_weight_file=self.lm_model.embed_weights_file,
                                                use_character_inputs=not self.token_lookup,
                                                max_batch_size=self.max_batch_size)
        q_lm_encoding = q_lm_model.get_ops()["lm_embeddings"]

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            c_lm_model = BidirectionalLanguageModel(self.lm_model.options_file, self.lm_model.weight_file,
                                                    input_tensors[self._context_char_ids_placeholder],
                                                    embedding_weight_file=self.lm_model.embed_weights_file,
                                                    use_character_inputs=not self.token_lookup,
                                                    max_batch_size=self._max_num_sentences)
            c_lm_encoding = c_lm_model.get_ops()["lm_embeddings"]

        if self.per_sentence:
            c_lm_encoding = tf.gather_nd(c_lm_encoding, input_tensors[self._context_sentence_ixs])

        q_mask = input_tensors[enc.question_len]
        c_mask = input_tensors[enc.context_len]

        q_embed = []
        c_embed = []

        if enc.question_chars in input_tensors:
            with tf.variable_scope("char-embed"):
                q, c = self.char_embed.embed(is_train,
                                             (input_tensors[enc.question_chars], input_tensors[enc.question_word_len]),
                                             (input_tensors[enc.context_chars], input_tensors[enc.context_word_len]))
            q_embed.append(q)
            c_embed.append(c)

        if enc.question_words in input_tensors:
            with tf.variable_scope("word-embed"):
                q, c = self.word_embed.embed(is_train,
                                            (input_tensors[enc.question_words], q_mask),
                                            (input_tensors[enc.context_words], c_mask))
            q_embed.append(q)
            c_embed.append(c)

        if enc.question_features in input_tensors:
            q_embed.append(input_tensors.get(enc.question_features))
            c_embed.append(input_tensors.get(enc.context_features))

        q_embed = tf.concat(q_embed, axis=2)
        c_embed = tf.concat(c_embed, axis=2)

        answer = [input_tensors[x] for x in enc.answer_encoder.get_placeholders()]
        return self._get_predictions_for(is_train, q_embed, q_mask, q_lm_encoding,
                                         c_embed, c_mask, c_lm_encoding, answer)

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask, question_lm,
                             context_embed, context_mask, context_lm,
                             answer) -> Prediction:
        raise NotImplementedError()

    def encode(self, batch: List[ContextAndQuestion], is_train: bool):
        if len(batch) > self.max_batch_size:
            raise ValueError("The model can only use a batch <= %d, but got %d" %
                             (self.max_batch_size, len(batch)))
        data = self.encoder.encode(batch, is_train)
        data[self._question_char_ids_placeholder] = self._batcher.batch_sentences([q.question for q in batch])
        data[self._is_train_placeholder] = is_train
        context_word_dim = data[self.encoder.context_words].shape[1]

        if not self.per_sentence:
            data[self._context_char_ids_placeholder] = \
                self._batcher.batch_sentences([x.get_context() for x in batch])
        else:
            data[self._context_char_ids_placeholder] = \
                self._batcher.batch_sentences(flatten_iterable([x.sentences for x in batch]))

            # Compute indices where context_sentence_ixs[sentence#, k, sentence_word#] = (batch#, k, batch_word#)
            # for each word. We use this to map the tokens built for the sentences back to
            # the format where sentences are flattened for each batch
            context_sentence_ixs = np.zeros((len(batch), 3, context_word_dim, 3), dtype=np.int32)
            total_sent_ix = 0
            for ix, point in enumerate(batch):
                word_ix = 0
                for sent_ix, sent in enumerate(point.sentences):
                    for w_ix in range(len(sent)):
                        for k in range(3):
                            context_sentence_ixs[ix, k, word_ix] = [total_sent_ix, k, w_ix]
                        word_ix += 1
                    total_sent_ix += 1
            data[self._context_sentence_ixs] = context_sentence_ixs
        return data


class AttentionWithElmo(ElmoQaModel):
    """ Elmo model that uses attention """

    def __init__(self,
                 encoder: DocumentAndQuestionEncoder,
                 lm_model: LanguageModel,
                 max_batch_size: int,
                 per_sentence: bool,
                 append_embed: bool,
                 append_before_atten: bool,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 lm_reduce_shared: Optional[Mapper],
                 lm_reduce: Optional[Mapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 predictor: Union[SequencePredictionLayer, AttentionPredictionLayer]):
        """
        :param per_sentence: Run the language model on single sentences at a time (not recommended)
        :param append_embed: Append the language model vector to the word embeddings
        :param append_before_atten: Append the language model vectors to the after pre-processing
            the original text
        """
        super().__init__(encoder, lm_model, per_sentence, max_batch_size, word_embed, char_embed)
        self.embed_mapper = embed_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.predictor = predictor
        self.append_embed = append_embed
        self.append_before_atten = append_before_atten
        self.lm_reduce = lm_reduce
        self.lm_reduce_shared = lm_reduce_shared

    def _with_lm(self, is_train, embed, lm, mask, question: bool, append: bool=False):
        kind_name = "question" if question else "context"
        prefix = "weight" if append else "weight_embed"

        if self.lm_reduce_shared is not None:
            with tf.variable_scope(prefix + "_lm", reuse=question):
                context_lm_r = self.lm_reduce_shared.apply(is_train, lm, mask)
        else:
            context_lm_r = lm

        if self.lm_reduce is not None:
            with tf.variable_scope(prefix + "_" + kind_name + "_lm"):
                context_lm_r = self.lm_reduce.apply(is_train, context_lm_r, mask)
        return tf.concat([embed, context_lm_r], axis=2)

    def _get_predictions_for(self, is_train,
                             question_embed, question_mask, question_lm,
                             context_embed, context_mask, context_lm,
                             answer) -> Prediction:
        if self.append_embed:
            context_rep = self._with_lm(is_train, context_embed, context_lm, context_mask, False)
            question_rep = self._with_lm(is_train, question_embed, question_lm, question_mask, True)
        else:
            context_rep, question_rep = context_embed, question_embed

        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.append_before_atten:
            context_rep = self._with_lm(is_train, context_rep, context_lm, context_mask, False, True)
            question_rep = self._with_lm(is_train, question_rep, question_lm, question_mask, True, True)

        with tf.variable_scope("build_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.match_encoder is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.match_encoder.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("predict"):
            if isinstance(self.predictor, AttentionPredictionLayer):
                return self.predictor.apply(is_train, context_rep, question_rep, answer, context_mask, question_mask)
            else:
                return self.predictor.apply(is_train, context_rep, answer, context_mask)