import runner
from data_processing.qa_data import FixedParagraphQaTrainingData, Batcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import StaticAttention, StaticAttentionSelf, AttentionEncoder
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, SequenceMapperSeq, DropoutLayer, FullyConnected, ConcatWithProduct, ChainBiMapper, \
    WithProjectedProduct, MapperSeq, ResidualLayer, WhitenLayer
from nn.recurrent_layers import BiRecurrentMapper, RecurrentEncoder, EncodeOverTime, GruCellSpec
from nn.similarity_layers import TriLinear
from nn.span_prediction import WithFixedContextPredictionLayer
from runner import SerializableOptimizer, TrainParams
from squad.build_dataset import SquadCorpus
from squad.squad_eval import BoundedSquadSpanEvaluator
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               train_aux_only=3,
                               num_epochs=20, log_period=30, eval_period=1200, save_period=1200,
                               eval_samples=dict(dev=None, train=8000))

    regularize = WhitenLayer("learn")
    dim = 50

    enc = SequenceMapperSeq(
        regularize,
        BiRecurrentMapper(GruCellSpec(dim)),
        regularize,
    )

    model = Attention(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        word_embed_layer=None,
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=15, init_scale=0.1),
            EncodeOverTime(RecurrentEncoder(GruCellSpec(50), None), mask=True),
            shared_parameters=True
        ),
        embed_mapper=enc,
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=StaticAttention(TriLinear(bias=True), ConcatWithProduct()),
        # attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(
            FullyConnected(dim*2, activation="tanh"),
            regularize,
            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
            FullyConnected(dim*2, activation="tanh"),
            regularize,
        ),
        predictor=WithFixedContextPredictionLayer(
            ResidualLayer(BiRecurrentMapper(GruCellSpec(dim))),
            AttentionEncoder(post_process=MapperSeq(FullyConnected(20, activation="tanh"), regularize)),
            WithProjectedProduct(include_tiled=True),
            ChainBiMapper(
                first_layer=BiRecurrentMapper(GruCellSpec(dim)),
                second_layer=BiRecurrentMapper(GruCellSpec(dim))
            )
        )
    )
    with open(__file__, "r") as f:
        notes = f.read()

    corpus = SquadCorpus()
    train_batching = Batcher(45, "bucket_context_words_3", True, False)
    eval_batching = Batcher(45, "context_words", False, False)
    data = FixedParagraphQaTrainingData(corpus, None, train_batching, eval_batching)

    eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17])]
    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()