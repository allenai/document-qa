import trainer
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from dataset import ClusteredBatcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import BiAttention, StaticAttentionSelf
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import SequenceMapperSeq, VariationalDropoutLayer, NullBiMapper, FullyConnected, ResidualLayer, \
    ConcatWithProduct, ChainBiMapper, MaxPool, Conv1d
from nn.recurrent_layers import CudnnGru, EncodeOverTime, FusedRecurrentEncoder
from nn.similarity_layers import TriLinear
from nn.span_prediction import BoundsPredictor
from squad.squad_data import SquadCorpus, DocumentQaTrainingData
from squad.squad_evaluators import BoundedSquadSpanEvaluator
from trainer import TrainParams, SerializableOptimizer
from utils import get_output_name_from_cli


def train_params():
    return TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                       ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                       num_epochs=30, log_period=30, eval_period=1200, save_period=1200,
                       eval_samples=dict(dev=None, train=8000))


def model():
    dim = 100
    recurrent_layer = CudnnGru(dim)

    return Attention(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        preprocess=None,
        word_embed_layer=None,
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.05, force_cpu=True),
            # EncodeOverTime(FusedRecurrentEncoder(60)),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(), True),
        match_encoder=SequenceMapperSeq(
            FullyConnected(dim * 2, activation="relu"),
            ResidualLayer(SequenceMapperSeq(
                VariationalDropoutLayer(0.8),
                recurrent_layer,
                VariationalDropoutLayer(0.8),
                StaticAttentionSelf(TriLinear(), ConcatWithProduct()),
                FullyConnected(dim * 2, activation="relu"),
            )),
            VariationalDropoutLayer(0.8),
        ),
        predictor=BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            )
        )
    )


def main(mode="paragraph"):
    out = get_output_name_from_cli()

    corpus = SquadCorpus()

    if mode == "paragraph":
        train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
        eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
        data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)
        eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17])]
    else:
        raise RuntimeError()

    with open(__file__, "r") as f:
        notes = f.read()

    trainer.start_training(data, model(), train_params(), eval, trainer.ModelDir(out), notes)


if __name__ == "__main__":
    main()