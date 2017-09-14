from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

import model_dir
import trainer
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from dataset import ClusteredBatcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder, DenseMultiSpanAnswerEncoder
from evaluator import LossEvaluator, SpanProbability, SpanEvaluator
from nn.attention import BiAttention, StaticAttentionSelf
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder, FixedWordEmbedderPlaceholders
from nn.layers import MaxPool, Conv1d, SequenceMapperSeq, VariationalDropoutLayer, NullBiMapper, FullyConnected, \
    ResidualLayer, ConcatWithProduct, ChainBiMapper, NullMapper
from nn.recurrent_layers import CudnnGru
from nn.similarity_layers import TriLinear
from nn.span_prediction import BoundsPredictor
from squad.squad_data import DocumentQaTrainingData, SquadCorpus
from trainer import TrainParams, SerializableOptimizer
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                       ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                       num_epochs=26, log_period=30, eval_period=1200, save_period=1200,
                       eval_samples=dict(dev=None, train=8000))

    recurrent_layer = CudnnGru(100, w_init=TruncatedNormal(stddev=0.05))
    # recurrent_layer = NullMapper()

    model = Attention(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        word_embed=FixedWordEmbedderPlaceholders(vec_name="glove.840B.300d", word_vec_init_scale=0,
                                                 placeholder_stddev=0.422, cpu=True, placeholder_flag=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        preprocess=None,
        word_embed_layer=None,
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(FullyConnected(100 * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(0.8),
                                            recurrent_layer,
                                            VariationalDropoutLayer(0.8),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(100 * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(0.8)),
        predictor=BoundsPredictor(ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ))
    )

    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = DocumentQaTrainingData(SquadCorpus(), None, train_batching, eval_batching)

    eval = [LossEvaluator(), SpanProbability(), SpanEvaluator(bound=[17], text_eval="squad")]
    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes, None)


if __name__ == "__main__":
    main()

