import argparse
from datetime import datetime

from data_processing.paragraph_qa import ContextLenKey, ContextLenBucketedKey

import trainer
from dataset import ClusteredBatcher
from evaluator import SpanProbability, LossEvaluator, SpanEvaluator
from experimental.niket_qa import NiketTrainingData
from trainer import ModelDir, TrainParams, SerializableOptimizer


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--init', help='model to initialize from')
    parser.add_argument('name', help='name of output to exmaine')
    args = parser.parse_args()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               max_checkpoints_to_keep=7, eval_at_zero=True,
                               eval_samples={},
                               num_epochs=10, log_period=10*2, eval_period=80, save_period=80)

    train_batching = ClusteredBatcher(30, ContextLenBucketedKey(3), True, True)
    eval_batching = ClusteredBatcher(50, ContextLenKey(), False, True)
    data = NiketTrainingData(None, train_batching, eval_batching)

    with open(__file__, "r") as f:
        notes = f.read()

    init_from = ModelDir(args.init).get_latest_checkpoint()
    model = ModelDir(args.init).get_model()
    # model.encoder.answer_encoder = DenseMultiSpanAnswerEncoder()
    # model.predictor.aggregate = "sum"

    trainer.start_training(data, model, train_params,
                           [LossEvaluator(), SpanEvaluator([1, 2, 3, 4, 5]), SpanProbability()],
                           ModelDir(args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")),
                           notes, init_from)


if __name__ == "__main__":
    main()