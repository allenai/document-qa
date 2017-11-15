from typing import List, Optional, Dict, Union, Set

import tensorflow as tf
from tensorflow import Tensor

from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ParagraphAndQuestionSpec
from docqa.encoder import DocumentAndQuestionEncoder
from docqa.model import Model, Prediction
from docqa.nn.embedder import WordEmbedder, CharWordEmbedder
from docqa.nn.layers import SequenceMapper, SequenceBiMapper, AttentionMapper, SequenceEncoder, \
    SequenceMapperWithContext, MapMulti, SequencePredictionLayer, AttentionPredictionLayer
from docqa.text_preprocessor import TextPreprocessor
from docqa.utils import ResourceLoader


class ParagraphQuestionModel(Model):
    """
    Base class for models that take paragraph/questions as input, handles embedding the
    text in a modular way.

    Its a bit of a hack, but at the moment we leave it up to the client to be aware of and use the `preprocessor`
    (if not None) before passing input to `encode`. This is in particular so the preprocessing can be done
    only once and before we sort/batch the input data
    """

    def __init__(self,
                 encoder: DocumentAndQuestionEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder] = None,
                 word_embed_layer: Optional[MapMulti] = None,
                 preprocessor: Optional[TextPreprocessor] = None):
        if word_embed is None and char_embed is None:
            raise ValueError()
        self.preprocessor = preprocessor
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.word_embed_layer = word_embed_layer
        self.encoder = encoder
        self._is_train_placeholder = None

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

    def set_input_spec(self, input_spec: ParagraphAndQuestionSpec, voc: Set[str],
                       word_vec_loader: ResourceLoader=None):
        if word_vec_loader is None:
            word_vec_loader = ResourceLoader()
        if self.word_embed is not None:
            self.word_embed.init(word_vec_loader, voc)
        if self.char_embed is not None:
            self.char_embed.embeder.init(word_vec_loader, voc)
        self.encoder.init(input_spec, True, self.word_embed,
                          None if self.char_embed is None else self.char_embed.embeder)
        self._is_train_placeholder = tf.placeholder(tf.bool, ())
        return self.encoder.get_placeholders()

    def get_placeholders(self):
        return self.encoder.get_placeholders() + [self._is_train_placeholder]

    def get_predictions_for(self, input_tensors: Dict[Tensor, Tensor]):
        is_train = input_tensors[self._is_train_placeholder]
        enc = self.encoder
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
            if self.word_embed_layer is not None:
                with tf.variable_scope("embed-map"):
                    q, c = self.word_embed_layer.apply(is_train,
                                                       (q, q_mask),
                                                       (c, c_mask))
            q_embed.append(q)
            c_embed.append(c)

        if enc.question_features in input_tensors:
            q_embed.append(input_tensors.get(enc.question_features))
            c_embed.append(input_tensors.get(enc.context_features))

        q_embed = tf.concat(q_embed, axis=2)
        c_embed = tf.concat(c_embed, axis=2)

        answer = [input_tensors[x] for x in enc.answer_encoder.get_placeholders()]
        return self._get_predictions_for(is_train, q_embed, q_mask, c_embed, c_mask, answer)

    def _get_predictions_for(self,
                             is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer) -> Prediction:
        raise NotImplemented()

    def encode(self, batch: List, is_train: bool):
        data = self.encoder.encode(batch, is_train)
        data[self._is_train_placeholder] = is_train
        return data

    def __getstate__(self):
        state = super().__getstate__()
        state["_is_train_placeholder"] = None
        return state

    def __setstate__(self, state):
        if "state" in state:
            if "preprocessor" not in state["state"]:
                state["state"]["preprocessor"] = None
        super().__setstate__(state)


class ContextOnly(ParagraphQuestionModel):

    def __init__(self, encoder: DocumentAndQuestionEncoder,
                 word_embed: Optional[WordEmbedder],
                 char_embed: Optional[CharWordEmbedder],
                 context_encoder: SequenceMapper,
                 prediction: SequencePredictionLayer):
        super().__init__(encoder, word_embed, char_embed)
        self.context_encoder = context_encoder
        self.prediction = prediction

    def _get_predictions_for(self, is_train,
                             question_embed, question_mask,
                             context_embed, context_mask,
                             answer) -> Prediction:
        with tf.variable_scope("encode"):
            self.context_encoder.apply(is_train, context_embed, context_mask)

        with tf.variable_scope("predict"):
            return self.prediction.apply(is_train, context_embed, answer, context_mask)


class Attention(ParagraphQuestionModel):
    """
    Model that encodes the question and context, then applies an attention mechanism
    between the two to produce a query-aware context representation, which is used to make a prediction.
    """
    def __init__(self, encoder: DocumentAndQuestionEncoder,
                 preprocess: Optional[TextPreprocessor],
                 word_embed: Optional[WordEmbedder],
                 word_embed_layer: Optional[MapMulti],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 question_mapper: Optional[SequenceMapper],
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 match_encoder: SequenceMapper,
                 predictor: Union[SequencePredictionLayer, AttentionPredictionLayer]):
        super().__init__(encoder, word_embed, char_embed, word_embed_layer, preprocess)
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.attention = attention
        self.match_encoder = match_encoder
        self.predictor = predictor

    def _get_predictions_for(self, is_train,
                             question_rep, question_mask,
                             context_rep, context_mask,
                             answer) -> Prediction:
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("buid_memories"):
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


class AttentionAndEncode(ParagraphQuestionModel):

    def __init__(self, encoder: DocumentAndQuestionEncoder,
                 word_embed: Optional[WordEmbedder],
                 word_embed_layer: Optional[MapMulti],
                 char_embed: Optional[CharWordEmbedder],
                 embed_mapper: Optional[SequenceMapper],
                 question_mapper: Optional[SequenceMapper],
                 question_encoder: SequenceEncoder,
                 context_mapper: Optional[SequenceMapper],
                 memory_builder: SequenceBiMapper,
                 attention: AttentionMapper,
                 post_attention_mapper: Optional[SequenceMapper],
                 contextual_mapper: SequenceMapperWithContext,
                 post_context_mapper: Optional[SequenceMapper],
                 predictor: SequencePredictionLayer):
        super().__init__(encoder, word_embed, char_embed, word_embed_layer)
        self.question_encoder = question_encoder
        self.embed_mapper = embed_mapper
        self.question_mapper = question_mapper
        self.context_mapper = context_mapper
        self.memory_builder = memory_builder
        self.contextual_mapper = contextual_mapper
        self.attention = attention
        self.post_attention_mapper = post_attention_mapper
        self.post_context_mapper = post_context_mapper
        self.predictor = predictor

    def _get_predictions_for(self, is_train,
                             question_rep, question_mask,
                             context_rep, context_mask,
                             answer) -> Prediction:
        if self.embed_mapper is not None:
            with tf.variable_scope("map_embed"):
                context_rep = self.embed_mapper.apply(is_train, context_rep, context_mask)
            with tf.variable_scope("map_embed", reuse=True):
                question_rep = self.embed_mapper.apply(is_train, question_rep, question_mask)

        if self.question_mapper is not None:
            with tf.variable_scope("map_question"):
                question_rep = self.question_mapper.apply(is_train, question_rep, question_mask)

        if self.context_mapper is not None:
            with tf.variable_scope("map_context"):
                context_rep = self.context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("build_memories"):
            keys, memories = self.memory_builder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("apply_attention"):
            context_rep = self.attention.apply(is_train, context_rep, keys, memories, context_mask, question_mask)

        if self.post_attention_mapper is not None:
            with tf.variable_scope("process_attention"):
                context_rep = self.post_attention_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("encode_question"):
            question_encoded = self.question_encoder.apply(is_train, question_rep, question_mask)

        with tf.variable_scope("map_with_context"):
            context_rep = self.contextual_mapper.apply(is_train, context_rep, question_encoded, context_mask)

        if self.post_context_mapper is not None:
            with tf.variable_scope("process_context_mapped"):
                context_rep = self.post_context_mapper.apply(is_train, context_rep, context_mask)

        with tf.variable_scope("predict"):
            return self.predictor.apply(is_train, context_rep, answer, context_mask)
