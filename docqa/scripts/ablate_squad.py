import argparse
from datetime import datetime

from docqa import model_dir
from docqa import trainer
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphSetsBuilder, StratifyParagraphsBuilder, \
    RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ContextLenBucketedKey, ContextLenKey
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import ClusteredBatcher
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator, SpanEvaluator
from docqa.scripts.ablate_triviaqa import get_model
from docqa.squad.squad_data import SquadCorpus, DocumentQaTrainingData
from docqa.squad.squad_document_qa import SquadTfIdfRanker
from docqa.text_preprocessor import WithIndicators
from docqa.trainer import TrainParams, SerializableOptimizer


def train_params(n_epochs):
    return TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                       ema=0.999, max_checkpoints_to_keep=3, async_encoding=10,
                       num_epochs=n_epochs, log_period=30, eval_period=1200, save_period=1200,
                       eval_samples=dict(dev=None, train=8000))


def main():
    parser = argparse.ArgumentParser(description='Train a model on document-level SQuAD')
    parser.add_argument('mode', choices=["paragraph", "confidence", "shared-norm", "merge", "sigmoid"])
    parser.add_argument("name", help="Output directory")
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
        # Run in the "standard" known-paragraph setting
        if model.preprocessor is not None:
            raise NotImplementedError()
        n_epochs = 26

        train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
        eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
        data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)
        eval = [LossEvaluator(), SpanEvaluator(bound=[17], text_eval="squad")]
    else:
        eval_set_mode = {
            "confidence": "flatten",
            "sigmoid": "flatten",
            "shared-norm": "group",
            "merge": "merge"}[mode]
        eval_dataset = RandomParagraphSetDatasetBuilder(100, eval_set_mode, True, 0)

        if mode == "confidence" or mode == "sigmoid":
            if mode == "sigmoid":
                # needs to be trained for a really long time for reasons unknown, even this might be too small
                n_epochs = 100
            else:
                n_epochs = 50  # more epochs since we only "see" the label very other epoch-osh
            train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
            data = PreprocessedData(
                SquadCorpus(),
                SquadTfIdfRanker(NltkPlusStopWords(True), 4, True, model.preprocessor),
                StratifyParagraphsBuilder(train_batching, 1),
                eval_dataset,
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

    params = train_params(n_epochs)
    if mode == "paragraph":
        params.best_weights = ("dev", "b17/text-f1")

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes)


if __name__ == "__main__":
    main()