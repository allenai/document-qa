import trainer
from data_processing.qa_data import FixedParagraphQaTrainingData, Batcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator, SentenceSpanEvaluator, BoundedSpanEvaluator
from nn.attention import StaticAttentionSelf, BiAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, SequenceMapperSeq, DropoutLayer, ChainBiMapper, FullyConnected, ConcatWithProduct
from nn.recurrent_layers import BiRecurrentMapper, RecurrentEncoder, EncodeOverTime, GruCellSpec
from nn.similarity_layers import TriLinear
from nn.span_prediction import BoundsPredictor
from trainer import SerializableOptimizer, TrainParams
from squad.squad import SquadCorpus
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=20, log_period=30, eval_period=1200, save_period=1200,
                               eval_samples=dict(dev=None, train=8000))

    enc = SequenceMapperSeq(
        DropoutLayer(0.8),
        BiRecurrentMapper(GruCellSpec(80)),
        DropoutLayer(0.8),
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
        # attention=StaticAttention(TriLinear(bias=True), ConcatWithProduct()),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(
            FullyConnected(160, activation="tanh"),
            DropoutLayer(0.8),
            BiRecurrentMapper(GruCellSpec(80)),
            DropoutLayer(0.8),
            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
            FullyConnected(160, activation="tanh"),
            DropoutLayer(0.8),
        ),
        predictor=BoundsPredictor(ChainBiMapper(
            first_layer=BiRecurrentMapper(GruCellSpec(80)),
            second_layer=BiRecurrentMapper(GruCellSpec(80)),
        ))
    )
    with open(__file__, "r") as f:
        notes = f.read()

    corpus = SquadCorpus()
    train_batching = Batcher(45, "bucket_context_words_3", True, False)
    eval_batching = Batcher(45, "context_words", False, False)
    data = FixedParagraphQaTrainingData(corpus, None, train_batching, eval_batching)

    eval = [LossEvaluator(), BoundedSpanEvaluator(bound=[17]), SentenceSpanEvaluator()]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()
