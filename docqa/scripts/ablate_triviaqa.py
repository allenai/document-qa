import argparse
from datetime import datetime
from typing import Optional

from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ParagraphAndQuestionsBuilder, ContextLenKey, ContextLenBucketedKey
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import ClusteredBatcher
from docqa.doc_qa_models import Attention
from docqa.encoder import DocumentAndQuestionEncoder, DenseMultiSpanAnswerEncoder, GroupedSpanAnswerEncoder
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator, SpanEvaluator
from docqa.nn.attention import BiAttention, AttentionEncoder, StaticAttentionSelf
from docqa.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from docqa.nn.layers import NullBiMapper, SequenceMapperSeq, Conv1d, FullyConnected, \
    ChainBiMapper, ConcatWithProduct, ResidualLayer, VariationalDropoutLayer, MaxPool
from docqa.nn.recurrent_layers import CudnnGru
from docqa.nn.similarity_layers import TriLinear
from docqa.nn.span_prediction import ConfidencePredictor, BoundsPredictor, IndependentBoundsGrouped, \
    IndependentBoundsSigmoidLoss
from docqa.text_preprocessor import WithIndicators, TextPreprocessor
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset
from docqa.triviaqa.training_data import ExtractSingleParagraph, ExtractMultiParagraphs


def get_triviaqa_train_params(n_epochs, n_dev, n_train):
    return TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1)),
        num_epochs=n_epochs, ema=0.9999, max_checkpoints_to_keep=2,
        async_encoding=10, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=n_dev, train=n_train))


def get_model(char_th: int, dim: int, mode: str, preprocess: Optional[TextPreprocessor]):
    recurrent_layer = CudnnGru(dim, w_init=TruncatedNormal(stddev=0.05))

    if mode.startswith("shared-norm"):
        answer_encoder = GroupedSpanAnswerEncoder()
        predictor = BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            ),
            span_predictor=IndependentBoundsGrouped(aggregate="sum")
        )
    elif mode == "confidence":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = ConfidencePredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer,
            ),
            AttentionEncoder(),
            FullyConnected(80, activation="tanh"),
            aggregate="sum"
        )
    elif mode == "sigmoid":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = BoundsPredictor(
            ChainBiMapper(
                first_layer=recurrent_layer,
                second_layer=recurrent_layer
            ),
            span_predictor=IndependentBoundsSigmoidLoss()
        )
    elif mode == "paragraph" or mode == "merge":
        answer_encoder = DenseMultiSpanAnswerEncoder()
        predictor = BoundsPredictor(ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ))
    else:
        raise NotImplementedError(mode)

    return Attention(
        encoder=DocumentAndQuestionEncoder(answer_encoder),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=char_th, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        preprocess=preprocess,
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
        match_encoder=SequenceMapperSeq(FullyConnected(dim * 2, activation="relu"),
                                        ResidualLayer(SequenceMapperSeq(
                                            VariationalDropoutLayer(0.8),
                                            recurrent_layer,
                                            VariationalDropoutLayer(0.8),
                                            StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                            FullyConnected(dim * 2, activation="relu"),
                                        )),
                                        VariationalDropoutLayer(0.8)),
        predictor=predictor
    )


def main():
    parser = argparse.ArgumentParser(description='Train a model on TriviaQA web')
    parser.add_argument('mode', choices=["paragraph-level", "confidence", "merge",
                                         "shared-norm", "sigmoid", "shared-norm-600"])
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with")
    args = parser.parse_args()
    mode = args.mode

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    model = get_model(100, 140, mode, WithIndicators())

    stop = NltkPlusStopWords(True)

    if mode == "paragraph-level":
        extract = ExtractSingleParagraph(MergeParagraphs(400), TopTfIdf(stop, 1), model.preprocessor, intern=True)
    elif mode == "shared-norm-600":
        extract = ExtractMultiParagraphs(MergeParagraphs(600), TopTfIdf(stop, 4), model.preprocessor, intern=True)
    else:
        extract = ExtractMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 4), model.preprocessor, intern=True)

    if mode == "paragraph-level":
        n_epochs = 16
        train = ParagraphAndQuestionsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True))
        test = ParagraphAndQuestionsBuilder(ClusteredBatcher(60, ContextLenKey(), False))
        n_dev, n_train = 21000, 12000
        eval = [LossEvaluator(), SpanEvaluator([4, 8], "triviaqa")]
    else:
        eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge")]
        # we sample two paragraphs per a (question, doc) pair, so evaluate on fewer questions
        n_dev, n_train = 15000, 8000

        if mode == "confidence" or mode == "sigmoid":
            if mode == "sigmoid":
                # Trains very slowly, do this at your own risk
                n_epochs = 71
            else:
                n_epochs = 28
            test = RandomParagraphSetDatasetBuilder(120, "flatten", True, 1)
            train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), 0, 1)
        else:
            n_epochs = 14
            test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, 1)
            train = StratifyParagraphSetsBuilder(35, mode == "merge", True, 1)

    data = TriviaQaWebDataset()

    params = get_triviaqa_train_params(n_epochs, n_dev, n_train)

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(args.n_processes, 1000)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "*" * 10 + "\nMode: " + args.mode + "\n" + "*"*10 + "\n" + notes

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()