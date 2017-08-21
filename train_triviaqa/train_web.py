from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.paragraph_qa import ContextLenKey, ContextLenBucketedKey
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.text_utils import NltkPlusStopWords
from dataset import ListBatcher, ClusteredBatcher
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import BiAttention, AttentionEncoder, StaticAttentionSelf
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, ReduceLayer, Conv1d, HighwayLayer, FullyConnected, \
    ChainBiMapper, DropoutLayer, ConcatWithProduct, ResidualLayer, WithProjectedProduct, MapperSeq, ChainConcat
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec, BiDirectionalFusedLstm, RecurrentEncoder, \
    EncodeOverTime, FusedRecurrentEncoder, CudnnGru, CudnnLstm
from nn.similarity_layers import TriLinear
from nn.span_prediction import ConfidencePredictor, BoundsPredictor, WithFixedContextPredictionLayer
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.lazy_data import LazyRandomParagraphBuilder
from trivia_qa.triviaqa_evaluators import ConfidenceEvaluator, TfTriviaQaBoundedSpanEvaluator
from trivia_qa.triviaqa_training_data import InMemoryWebQuestionBuilder, ExtractPrecomputedParagraph, \
    ExtractSingleParagraph
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=15, ema=0.999, max_checkpoints_to_keep=2,
                               async_encoding=10,
                               log_period=30, eval_period=1400, save_period=1400,
                               eval_samples=dict(dev=18000, train=10000))

    recurrent_layer = SequenceMapperSeq(DropoutLayer(0.8), CudnnLstm(100))
    model = Attention(
        encoder=DocumentAndQuestionEncoder(DenseMultiSpanAnswerEncoder()),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            embedder=LearnedCharEmbedder(16, 49, 8),
            layer=ReduceLayer("max", Conv1d(100, 5, 0.8), mask=False),
            shared_parameters=True
        ),
        word_embed_layer=None,
        embed_mapper=SequenceMapperSeq(
            HighwayLayer(activation="relu"), HighwayLayer(activation="relu"),
            recurrent_layer),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=NullMapper(),
        predictor= BoundsPredictor(
            ChainConcat(
                start_layer=SequenceMapperSeq(
                    recurrent_layer,
                    recurrent_layer),
                end_layer=recurrent_layer
            )
        )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = ClusteredBatcher(60, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, False)
    stop = NltkPlusStopWords()
    data = PreprocessedData(TriviaQaWebDataset(),
                            ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), intern=True),
                            InMemoryWebQuestionBuilder(train_batching, eval_batching),
                            eval_on_verified=False
                            )
    eval = [LossEvaluator(), TfTriviaQaBoundedSpanEvaluator([4, 8])]
    data.load_preprocess("triviaqa-web-merge400-tfidf1.pkl.gz")
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes, False)



if __name__ == "__main__":
    main()