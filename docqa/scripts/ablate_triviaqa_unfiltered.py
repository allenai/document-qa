import argparse
from datetime import datetime

from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ContextLenBucketedKey
from docqa.dataset import ClusteredBatcher
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from docqa.scripts.ablate_triviaqa import get_model
from docqa.text_preprocessor import WithIndicators
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion


def main():
    parser = argparse.ArgumentParser(description='Train a model on TriviaQA unfiltered')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("-t", "--n_tokens", default=400, type=int,
                        help="Paragraph size")
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with"
                        )
    args = parser.parse_args()
    mode = args.mode

    out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")

    model = get_model(100, 140, mode, WithIndicators())

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(args.n_tokens), ShallowOpenWebRanker(16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge", per_doc=False)]
    oversample = [1] * 4

    if mode == "paragraph":
        n_epochs = 120
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True),
                                          oversample,  only_answers=True)
    elif mode == "confidence" or mode == "sigmoid":
        if mode == "sigmoid":
            n_epochs = 640
        else:
            n_epochs = 160
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), oversample)
    else:
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)
        train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)

    data = TriviaQaOpenDataset()

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