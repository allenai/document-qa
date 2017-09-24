import argparse
from datetime import datetime

import model_dir
import trainer
from data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ContextLenBucketedKey
from dataset import ClusteredBatcher
from evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from scripts.ablate_triviaqa import get_model
from text_preprocessor import WithIndicators
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.training_data import ExtractMultiParagraphsPerQuestion


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm", "sigmoid", "paragraph"])
    parser.add_argument("name")
    parser.add_argument('-n', '--n_processes', type=int, default=2)
    args = parser.parse_args()
    mode = args.mode

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    model = get_model(100, 140, mode, WithIndicators())

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(400), ShallowOpenWebRanker(16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge")]
    oversample = [1] * 4

    if mode == "paragraph":
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True),
                                          oversample,  only_answers=True)
    elif mode == "confidence" or mode == "sigmoid":
        if mode == "sigmoid":
            n_epochs = 320
        else:
            n_epochs = 160
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), oversample)
    else:
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)
        train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)

    data = TriviaQaWebDataset()

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=10, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=None, train=6000)
    )

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(args.n_processes, 1000)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "Mode: " + args.mode + "\n" + notes

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()