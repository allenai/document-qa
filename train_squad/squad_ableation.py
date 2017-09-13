import argparse
from datetime import datetime

import model_dir
import trainer
from configurable import Configurable
from data_processing.multi_paragraph_qa import StratifyParagraphSetsBuilder, StratifyParagraphsBuilder, \
    RandomParagraphsBuilder, RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from doc_qa_models import Attention, ContextOnly
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder, DenseMultiSpanAnswerEncoder
from evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from nn.attention import BiAttention, StaticAttentionSelf, AttentionEncoder, NullAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import SequenceMapperSeq, VariationalDropoutLayer, NullBiMapper, FullyConnected, ResidualLayer, \
    ConcatWithProduct, ChainBiMapper, MaxPool, Conv1d, NullMapper, IndependentBiMapper, SequencePredictionLayer, \
    get_keras_initialization, SequenceMapper
from nn.recurrent_layers import CudnnGru, EncodeOverTime, FusedRecurrentEncoder, CudnnLstm
from nn.similarity_layers import TriLinear
from nn.span_prediction import BoundsPredictor, ConfidencePredictor, IndependentBounds, BoundaryPrediction
from squad.squad_data import SquadCorpus, DocumentQaTrainingData
from squad.squad_document_qa import SquadTfIdfRanker
from squad.squad_evaluators import BoundedSquadSpanEvaluator, SquadConfidenceEvaluator
from text_preprocessor import WithIndicators
from train_triviaqa.train_ours import get_model
from trainer import TrainParams, SerializableOptimizer
from utils import get_output_name_from_cli


def train_params(n_epochs):
    return TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                       ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                       num_epochs=n_epochs, log_period=30, eval_period=1200, save_period=1200,
                       eval_samples=dict(dev=None, train=8000))


def main(mode="paragraph"):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=["paragraph", "confidence", "shared-norm", "merge", "sigmoid"])
    parser.add_argument("name")
    args = parser.parse_args()
    mode = args.mode
    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    corpus = SquadCorpus()
    if mode == "merge":
        # Adds paragraph start tokens, since we will be concatenating paragraphs together
        pre = WithIndicators(True, para_tokens=False, doc_start_token=False)
    else:
        pre = None

    model = get_model(50, 100, args.mode, pre)

    if mode == "paragraph":
        if model.preprocessor is not None:
            raise NotImplementedError()
        n_epochs = 26
        train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
        eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
        data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)
        eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17])]
    else:
        eval_set_mode = {
            "confidence": "flatten",
            "sigmoid": "flatten",
            "shared-norm": "group",
            "merge": "merge"}[mode]
        eval_dataset = RandomParagraphSetDatasetBuilder(100, eval_set_mode, True, 0)

        if mode == "confidence" or mode == "sigmoid":
            n_epochs = 50  # lmore epochs since we only "see" the label point very other epoch-osh
            train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
            data = PreprocessedData(
                SquadCorpus(),
                SquadTfIdfRanker(NltkPlusStopWords(True), 4, True, model.preprocessor),
                StratifyParagraphsBuilder(train_batching, 0, 1),
                eval_dataset,
                sample_dev=15, sample=50,
                eval_on_verified=False,
            )
        else:
            n_epochs = 26
            data = PreprocessedData(
                SquadCorpus(),
                SquadTfIdfRanker(NltkPlusStopWords(True), 4, True, model.preprocessor),
                StratifyParagraphSetsBuilder(25, args.mode == "merge", True, 1),
                eval_dataset,
                eval_on_verified=False,
            )

        eval = [LossEvaluator(), MultiParagraphSpanEvaluator(17, "squad")]
        data.preprocess(1)

    with open(__file__, "r") as f:
        notes = f.read()
        notes = args.mode + "\n" + notes

    trainer.start_training(data, model, train_params(n_epochs), eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()