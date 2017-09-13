import argparse
from datetime import datetime

import model_dir
import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_training_data import ParagraphAndQuestionsBuilder, ContextLenKey, ContextLenBucketedKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ClusteredBatcher
from evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from text_preprocessor import WithIndicators
from train_triviaqa.train_ours import get_model
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.training_data import ExtractSingleParagraph, ExtractMultiParagraphs, ExtractMultiParagraphsPerQuestion


def get_triviaqa_train_params(n_epochs, n_dev, n_train):
    return TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=10, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=n_dev, train=n_train))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm", "sigmoid"])
    parser.add_argument("name")
    parser.add_argument('-n', '--n_processes', type=int, default=2)
    args = parser.parse_args()
    mode = args.mode

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    model = get_model(100, 140, mode, WithIndicators())

    stop = NltkPlusStopWords(True)

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(400), TopTfIdf(stop, 16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge")]
    # we sample two paragraphs per a (question, doc) pair, so evaluate on fewer questions
    n_dev, n_train = 15000, 8000

    if mode == "confidence" or mode == "sigmoid":
        n_epochs = 24  # only see one paragraph an epoch, so train for more epochs
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, 1)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), 0, 1)
    else:
        n_epochs = 16
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, 1)
        train = StratifyParagraphSetsBuilder(35, mode == "merge", True, 1)

    data = TriviaQaWebDataset()

    params = get_triviaqa_train_params(n_epochs, n_dev, n_train)

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(args.n_processes, 1000)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "Mode: " + args.mode + "\n" + "Dataset: " + args.dataset + "\n" + notes

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()