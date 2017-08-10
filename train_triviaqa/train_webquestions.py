import runner
from data_processing.document_splitter import Truncate, MergeParagraphs, TopTfIdf
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import NltkPlusStopWords
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import StaticAttention, StaticAttentionSelf, AttentionEncoder, BiAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, SequenceMapperSeq, DropoutLayer, FullyConnected, ChainBiMapper, \
    ConcatWithProductProj, ConcatWithProduct, ResidualLayer, WithProjectedProduct, MapperSeq, HighwayLayer, ReduceLayer, \
    Conv1d
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec, RecurrentEncoder, EncodeOverTime, GruCellSpec
from nn.similarity_layers import DotProductProject, TriLinear, SimilaritySum
from nn.span_prediction import BoundsPredictor, WithFixedContextPredictionLayer
from runner import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.lazy_data import LazyRandomParagraphBuilder
from trivia_qa.triviaqa_evaluators import TfTriviaQaBoundedSpanEvaluator
from trivia_qa.triviaqa_training_data import InMemoryWebQuestionBuilder, ExtractSingleParagraph
from utils import get_output_name_from_cli



def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               ema=0.999, max_checkpoints_to_keep=3,
                               num_epochs=20, log_period=30, eval_period=1200, save_period=1200,
                               eval_samples=dict(dev=10000, train=7000))
    dropout = 0.8

    model = Attention(
        encoder=DocumentAndQuestionEncoder(DenseMultiSpanAnswerEncoder()),
        word_embed_layer=None,
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=25, init_scale=0.1),
            EncodeOverTime(RecurrentEncoder(GruCellSpec(80), None), mask=True),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            DropoutLayer(0.9),
            BiRecurrentMapper(GruCellSpec(100)),
            DropoutLayer(dropout),
        ),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=SequenceMapperSeq(
            FullyConnected(200, activation="tanh"),
            DropoutLayer(dropout),
            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
            FullyConnected(200, activation="tanh"),
            DropoutLayer(dropout),
        ),
        predictor=BoundsPredictor(
            ChainBiMapper(
                first_layer=BiRecurrentMapper(GruCellSpec(100)),
                second_layer=BiRecurrentMapper(GruCellSpec(100))
            ),
            aggregate="sum"
        )
    )
    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = Batcher(60, "bucket_context_words_3", True, False)
    eval_batching = Batcher(60, "context_words", False, False)
    stop = NltkPlusStopWords()
    data = PreprocessedData(TriviaQaWebDataset(),
                            ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), True),
                            InMemoryWebQuestionBuilder(train_batching, eval_batching),
                            # sample_dev=100, sample=100, eval_on_verified=False
                            )
    data.preprocess(4, 1000)

    eval = [LossEvaluator(), TfTriviaQaBoundedSpanEvaluator([4, 8])]
    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()