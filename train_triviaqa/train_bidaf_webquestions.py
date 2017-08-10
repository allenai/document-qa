import runner
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import NltkPlusStopWords
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.attention import BiAttention, AttentionEncoder
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, ReduceLayer, Conv1d, HighwayLayer, FullyConnected
from nn.prediction_layers import ChainConcat
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec
from nn.similarity_layers import TriLinear
from nn.span_prediction import ConfidencePredictor, BoundsPredictor
from runner import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.lazy_data import LazyRandomParagraphBuilder
from trivia_qa.triviaqa_evaluators import ConfidenceEvaluator, TfTriviaQaBoundedSpanEvaluator
from trivia_qa.triviaqa_training_data import InMemoryWebQuestionBuilder, ExtractPrecomputedParagraph, \
    ExtractSingleParagraph
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=8, ema=0.999, max_checkpoints_to_keep=2,
                               log_period=30, eval_period=1000, save_period=1000,
                               eval_samples={"train": 7000, "dev": 10000, "verified-dev": None})

    model = Attention(
        encoder=DocumentAndQuestionEncoder(DenseMultiSpanAnswerEncoder()),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            embedder=LearnedCharEmbedder(16, 49, 8),
            layer=ReduceLayer("max", Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        word_embed_layer=None,
        embed_mapper=SequenceMapperSeq(
            HighwayLayer(activation="relu"),
            HighwayLayer(activation="relu"),
            BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=NullMapper(),
        predictor=BoundsPredictor(
            ChainConcat(
                start_layer=SequenceMapperSeq(
                    BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8)),
                    BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))),
                end_layer=BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8)),
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
                            ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), intern=True),
                            InMemoryWebQuestionBuilder(train_batching, eval_batching),
                            # sample_dev=100, sample=100, eval_on_verified=False
                            )

    eval = [LossEvaluator(), TfTriviaQaBoundedSpanEvaluator([4, 8])]
    data.preprocess(4, 1000)
    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()