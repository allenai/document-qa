from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal
from trivia_qa.triviaqa_evaluators import ConfidenceEvaluator

import model_dir
import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.multi_paragraph_qa import RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.text_utils import NltkPlusStopWords
from doc_qa_models import Attention
from encoder import GroupedSpanAnswerEncoder, CheatingEncoder
from evaluator import LossEvaluator
from nn.attention import NullAttention
from nn.embedder import FixedWordEmbedder
from nn.layers import NullBiMapper, NullMapper, FullyConnected
from nn.recurrent_layers import CudnnGru
from nn.span_prediction import BoundsPredictor, IndependentBoundsGrouped
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.training_data import ExtractMultiParagraphs
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(
                               SerializableOptimizer("Adadelta", dict(learning_rate=1)),
                               # SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=15, ema=0.999, max_checkpoints_to_keep=2,
                               async_encoding=10,
                               log_period=10, eval_period=40, save_period=1800,
                               eval_samples=dict(dev=210, train=120))
    dim = 140
    recurrent_layer = CudnnGru(dim, w_init=TruncatedNormal())

    model = Attention(
        # encoder=DocumentAndQuestionEncoder(GroupedSpanAnswerEncoder()),
        encoder=CheatingEncoder(GroupedSpanAnswerEncoder()),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False,
                                     cpu=True),
        # char_embed=CharWordEmbedder(
        #     LearnedCharEmbedder(word_size_th=14, char_th=100, char_dim=20, init_scale=0.05, force_cpu=True),
        #     EncodeOverTime(FusedRecurrentEncoder(60), mask=False),
        #     shared_parameters=True
        # ),
        char_embed=None,
        word_embed_layer=None,
        embed_mapper=FullyConnected(10),
        # SequenceMapperSeq(
        #     VariationalDropoutLayer(0.8),

            # VariationalDropoutLayer(0.8),
        # ),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=NullAttention(),
        match_encoder=NullMapper(),
        # attention=BiAttention(TriLinear(bias=True), True),
        # match_encoder=SequenceMapperSeq(FullyConnected(dim*2, activation="relu"),
                                        # ResidualLayer(SequenceMapperSeq(
                                        #     VariationalDropoutLayer(0.8),
                                        #     recurrent_layer,
                                        #     VariationalDropoutLayer(0.8),
                                        #     StaticAttentionSelf(TriLinear(bias=True), ConcatWithProduct()),
                                        #     FullyConnected(dim * 2, activation="relu")
                                        # )),
                                        # VariationalDropoutLayer(0.8)),
        predictor=BoundsPredictor(
            NullBiMapper(),
            # ChainBiMapper(
            #     first_layer=recurrent_layer,
            #     second_layer=recurrent_layer,
            # ),
            span_predictor=IndependentBoundsGrouped()
        )
        # predictor = ConfidencePredictor(
        #     NullBiMapper(),
        #     AttentionEncoder(),
        #     FullyConnected(80, activation="tanh"),
        #     aggregate="sum"
        # )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    stop = NltkPlusStopWords(True)
    prep = ExtractMultiParagraphs(MergeParagraphs(400), TopTfIdf(stop, 4),
                                  intern=True, require_an_answer=True)
    builder = RandomParagraphSetDatasetBuilder(10, 10, 2, True)
    # train_batching = ClusteredBatcher(20, ContextLenBucketedKey(3), True, False)
    # eval_batching = ClusteredBatcher(20, ContextLenKey(), False, False)
    # builder = RandomParagraphDatasetBuilder(train_batching, eval_batching, 0.5, True)
    # builder = ConcatParagraphDatasetBuilder(train_batching, eval_batching, False)
    data = PreprocessedData(TriviaQaWebDataset(), prep, builder, eval_on_verified=False,
                            sample=200, sample_dev=100
                            )
    eval = [LossEvaluator(), ConfidenceEvaluator(8)]
    data.preprocess(6, 1000)
    # data.cache_preprocess("tmp.pkl.gz")
    # data.load_preprocess("tfidf-top4-in-mem.pkl.gz")
    trainer.start_training(data, model, train_params, eval, model_dir.ModelDir(out), notes, None)


if __name__ == "__main__":
    main()