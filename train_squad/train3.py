from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

import trainer
from data_processing.paragraph_qa import ContextLenKey, ContextLenBucketedKey, DocumentQaTrainingData
from dataset import ClusteredBatcher, ListBatcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import StaticAttention, StaticAttentionSelf, AttentionEncoder
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder, DropNames
from nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, DropoutLayer, FullyConnected, ChainBiMapper, \
    ConcatWithProduct, WithProjectedProduct, \
    MapperSeq, ResidualLayer, MergeWith, VariationalDropoutLayer
from nn.recurrent_layers import BiRecurrentMapper, RecurrentEncoder, EncodeOverTime, GruCellSpec, \
    BiDirectionalFusedLstm, FusedRecurrentEncoder, BlockGruCellSpec, CudnnGru, CudnnLstm
from nn.similarity_layers import TriLinear
from nn.span_prediction import WithFixedContextPredictionLayer, SpanFromBoundsPredictor, BoundsPredictor
from squad.build_squad_dataset import SquadCorpus
from trainer import SerializableOptimizer, TrainParams


from squad.squad_eval import BoundedSquadSpanEvaluator
from utils import get_output_name_from_cli
from data_processing.text_utils import NameDetector


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                               num_epochs=30, log_period=30, eval_period=1200, save_period=1200,
                               eval_samples=dict(dev=None, train=8000))

    dim = 80
    recurrent_layer = CudnnGru(dim)

    model = Attention(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        word_embed_layer=None,
        # word_embed=DropNames(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False,
        #                      keep_probs=0.7, kind="shuffle"),
        word_embed=DropNames(selector=NameDetector(), vec_name="glove.840B.300d", word_vec_init_scale=0,
                             swapped_flag=False, learn_unk=False, keep_probs=0.6, kind="shuffle"),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=20, init_scale=0.1, force_cpu=True),
            EncodeOverTime(FusedRecurrentEncoder(60), mask=True),
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
        attention=StaticAttention(TriLinear(bias=True), ConcatWithProduct()),
        # attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(
            FullyConnected(dim * 2, activation="relu"),
            ResidualLayer(SequenceMapperSeq(
                VariationalDropoutLayer(0.8),
                recurrent_layer,
                VariationalDropoutLayer(0.8),
                StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                FullyConnected(dim * 2, activation="relu"),
            )),
            VariationalDropoutLayer(0.8),
        ),
        predictor=WithFixedContextPredictionLayer(
            ResidualLayer(recurrent_layer),
            AttentionEncoder(post_process=MapperSeq(FullyConnected(25, activation="tanh"), DropoutLayer(0.8))),
            WithProjectedProduct(include_tiled=True),
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            ),
        )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    corpus = SquadCorpus()
    train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)

    eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17])]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes)


if __name__ == "__main__":
    main()