import argparse
from datetime import datetime

import trainer
from data_processing.qa_training_data import ContextLenKey, ContextLenBucketedKey
from dataset import ClusteredBatcher
from encoder import DenseMultiSpanAnswerEncoder
from evaluator import LossEvaluator, ConfidenceSpanEvaluator
from experimental.niket_qa.build_niket_dataset import NiketTrainingData
from experimental.prediction_layers import ChainConcatConfidenceHack
from model_dir import ModelDir
from trainer import TrainParams, SerializableOptimizer


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--init', help='model to initialize from')
    parser.add_argument('name', help='name of output to exmaine')
    args = parser.parse_args()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               max_checkpoints_to_keep=7, eval_at_zero=True,
                               eval_samples={},
                               num_epochs=60, log_period=10*2, eval_period=80, save_period=80)

    train_batching = ClusteredBatcher(30, ContextLenBucketedKey(3), True, True)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, True)
    data = NiketTrainingData(None, train_batching, eval_batching, False)

    with open(__file__, "r") as f:
        notes = f.read()

    init_from = ModelDir(args.init).get_latest_checkpoint()
    model = ModelDir(args.init).get_model()

    # recurrent_layer = BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))
    model.encoder.answer_encoder = DenseMultiSpanAnswerEncoder()
    model.predictor = ChainConcatConfidenceHack(
        model.predictor.start_layer,
        model.predictor.end_layer,
    )
    #
    trainer.start_training(data, model, train_params,
                           [LossEvaluator(), ConfidenceSpanEvaluator([6, 12])],
                           ModelDir(args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")),
                           notes, init_from)


if __name__ == "__main__":
    main()