from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

import model_dir
import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.multi_paragraph_qa import RandomParagraphsBuilder, StratifyParagraphSetsBuilder
from data_processing.preprocessed_corpus import PreprocessedData

from data_processing.qa_training_data import ParagraphAndQuestionsBuilder, ContextLenBucketedKey, ContextLenKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ListBatcher, ClusteredBatcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder, SingleSpanAnswerEncoder, \
    PackedMultiSpanAnswerEncoder, GroupedSpanAnswerEncoder
from evaluator import LossEvaluator, SpanProbability
from nn.attention import BiAttention, AttentionEncoder, StaticAttentionSelf, StaticAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, ReduceLayer, Conv1d, HighwayLayer, FullyConnected, \
    ChainBiMapper, DropoutLayer, ConcatWithProduct, ResidualLayer, WithProjectedProduct, MapperSeq, ParametricRelu, \
    VariationalDropoutLayer, ChainConcat
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec, BiDirectionalFusedLstm, EncodeOverTime, \
    FusedRecurrentEncoder, CudnnLstm, CudnnGru
from nn.similarity_layers import TriLinear
from nn.span_prediction import ConfidencePredictor, BoundsPredictor, WithFixedContextPredictionLayer, \
    BoundedSpanPredictor, SpanFromVectorBound, IndependentBoundsSigmoidLoss, IndependentBoundsNoAnswerOption
from text_preprocessor import WithIndicators
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from experimental.lazy_data import LazyRandomParagraphBuilder
from trivia_qa.training_data import ExtractSingleParagraph, ExtractMultiParagraphs
from trivia_qa.triviaqa_evaluators import ConfidenceEvaluator, BoundedSpanEvaluator, TfTriviaQaBoundedSpanEvaluator
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(
                               SerializableOptimizer("Adadelta", dict(learning_rate=1)),
                               # SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=17, ema=0.999, max_checkpoints_to_keep=2,
                               async_encoding=10,
                               log_period=30, eval_period=1800, save_period=1800,
                               eval_samples=dict(dev=24000, train=14000))

    dim = 140
    pre = WithIndicators(remove_cross_answer=True)
    recurrent_layer = CudnnGru(dim, w_init=TruncatedNormal())
    model = Attention(
        encoder=DocumentAndQuestionEncoder(DenseMultiSpanAnswerEncoder()),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=100, char_dim=20, init_scale=0.05, force_cpu=True),
            EncodeOverTime(FusedRecurrentEncoder(60), mask=False),
            shared_parameters=True
        ),
        preprocess=pre,
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
        match_encoder=SequenceMapperSeq(FullyConnected(dim*2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(0.8),
                                            recurrent_layer,
                                            VariationalDropoutLayer(0.8),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(dim * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(0.8)),
        predictor=BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer,
            ),
            span_predictor=IndependentBoundsNoAnswerOption()
        )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    stop = NltkPlusStopWords(True)
    prep = ExtractMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 4),
                                  model.preprocessor, intern=True, require_an_answer=True)

    eval_batching = ClusteredBatcher(150, ContextLenKey(), False, False)
    eval_builder = RandomParagraphsBuilder(eval_batching, eval_batching, 0.5)
    train_builder = StratifyParagraphSetsBuilder(35, 35, True, True)

    data = PreprocessedData(TriviaQaWebDataset(), prep, train_builder, eval_builder, eval_on_verified=False)
    data.preprocess(6, 1000)
    eval = [LossEvaluator(), ConfidenceEvaluator(8)]
    trainer.start_training(data, model, train_params, eval, model_dir.ModelDir(out), notes, None)


if __name__ == "__main__":
    main()