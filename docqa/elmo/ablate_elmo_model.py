import argparse
from datetime import datetime

from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

from docqa import trainer
from docqa.data_processing.qa_training_data import ContextLenKey
from docqa.dataset import ClusteredBatcher
from docqa.encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from docqa.evaluator import LossEvaluator, SpanEvaluator
from docqa.elmo.elmo import ElmoLayer
from docqa.elmo.lm_qa_models import AttentionWithElmo, SquadContextConcatSkip
from docqa.model_dir import ModelDir
from docqa.nn.attention import BiAttention, StaticAttentionSelf
from docqa.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from docqa.nn.layers import FullyConnected, ChainBiMapper, NullBiMapper, MaxPool, Conv1d, SequenceMapperSeq, \
    VariationalDropoutLayer, ResidualLayer, ConcatWithProduct, MapperSeq, DropoutLayer
from docqa.nn.recurrent_layers import CudnnGru
from docqa.nn.similarity_layers import TriLinear
from docqa.nn.span_prediction import BoundsPredictor
from docqa.squad.squad_data import SquadCorpus, DocumentQaTrainingData


def main():
    parser = argparse.ArgumentParser("Train our ELMo model on SQuAD")
    parser.add_argument("output_dir")
    parser.add_argument("--dim", type=int, default=90)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--mode", choices=["input", "output", "both", "none"], default="both")
    parser.add_argument("--top_layer_only", action="store_true")
    args = parser.parse_args()

    out = args.output_dir + "-" + datetime.now().strftime("%m%d-%H%M%S")

    dim = args.dim
    recurrent_layer = CudnnGru(dim, w_init=TruncatedNormal(stddev=0.05))

    params = trainer.TrainParams(trainer.SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                                 ema=0.999, max_checkpoints_to_keep=2, async_encoding=10,
                                 num_epochs=24, log_period=30, eval_period=1200, save_period=1200,
                                 best_weights=("dev", "b17/text-f1"),
                                 eval_samples=dict(dev=None, train=8000))

    lm_reduce = MapperSeq(
        ElmoLayer(args.l2, layer_norm=False, top_layer_only=args.top_layer_only),
        DropoutLayer(0.5),
    )

    model = AttentionWithElmo(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        lm_model=SquadContextConcatSkip(),
        append_before_atten=(args.mode == "both" or args.mode == "output"),
        append_embed=(args.mode == "both" or args.mode == "input"),
        max_batch_size=128,
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False, cpu=True),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=49, char_dim=20, init_scale=0.05, force_cpu=True),
            MaxPool(Conv1d(100, 5, 0.8)),
            shared_parameters=True
        ),
        embed_mapper=SequenceMapperSeq(
            VariationalDropoutLayer(0.8),
            recurrent_layer,
            VariationalDropoutLayer(0.8),
        ),
        lm_reduce=None,
        lm_reduce_shared=lm_reduce,
        per_sentence=False,
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
        predictor = BoundsPredictor(ChainBiMapper(
            first_layer=recurrent_layer,
            second_layer=recurrent_layer
        ))
    )

    batcher = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = DocumentQaTrainingData(SquadCorpus(), None, batcher, batcher)

    with open(__file__, "r") as f:
        notes = f.read()
        notes = str(sorted(args.__dict__.items(), key=lambda x:x[0])) + "\n" + notes

    trainer.start_training(data, model, params,
                           [LossEvaluator(), SpanEvaluator(bound=[17], text_eval="squad")],
                           ModelDir(out), notes)

if __name__ == "__main__":
    main()