import argparse
from datetime import datetime

import model_dir
import trainer
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from doc_qa_models import Attention
from encoder import DenseMultiSpanAnswerEncoder, DocumentAndQuestionEncoder
from evaluator import LossEvaluator
from nn.attention import AttentionEncoder, BiAttention, StaticAttentionSelf
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import ChainBiMapper, FullyConnected, MaxPool, Conv1d, SequenceMapperSeq, VariationalDropoutLayer, \
    NullBiMapper, ResidualLayer, ConcatWithProduct
from nn.recurrent_layers import CudnnGru
from nn.similarity_layers import TriLinear
from nn.span_prediction import ConfidencePredictor
from squad.squad_data import SquadCorpus
from squad.squad_document_qa import SquadTfIdfRanker, StratifySquadParagraphBuilder, RandomSquadParagraphBuilder
from squad.squad_evaluators import SquadConfidenceEvaluator


def train_params(n_epochs):
    return TrainParams(SerializabæleOptimizer("Adadelta", dict(learning_rate=1.0)),
                       ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                       num_epochs=n_epochs, log_period=30, eval_period=1200, save_period=1200,
                       eval_samples=dict(dev=None, train=8000))


def model():
    dim = 100
    recurrent_layer = CudnnGru(dim)

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

    return Attention(
        encoder=DocumentAndQuestionEncoder(answer_encoder),
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
        predictor=predictor
    )


def main(mode="paragraph"):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("name")
    parser.add_argument('mode', choices=["paragraph", "confidence"])
    args = parser.parse_args()
    mode = args.mode
    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    corpus = SquadCorpus()

    n_epochs = 35
    train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = PreprocessedData(
        SquadCorpus(),
        SquadTfIdfRanker(NltkPlusStopWords(True), 4, True),
        StratifySquadParagraphBuilder(train_batching, eval_batching, 2),
        RandomSquadParagraphBuilder(train_batching, eval_batching, 0.5),
        eval_on_verified=False,
    )
    eval = [LossEvaluator(), SquadConfidenceEvaluator(17)]
    data.preprocess(1)

    with open(__file__, "r") as f:
        notes = f.read()
        notes = args.mode + "\n" + notes

    trainer.start_training(data, model(), train_params(n_epochs), eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()