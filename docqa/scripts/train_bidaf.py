from docqa.data_processing.paragraph_qa import ContextLenKey, ContextLenBucketedKey, DocumentQaTrainingData
from docqa.squad.squad_eval import SentenceSpanEvaluator, SquadSpanEvaluator, BoundedSquadSpanEvaluator

from docqa import model_dir
from docqa import trainer
from docqa.dataset import ClusteredBatcher
from docqa.doc_qa_models import Attention
from docqa.encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from docqa.evaluator import LossEvaluator
from docqa.nn.attention import BiAttention
from docqa.nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from docqa.nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, ReduceLayer, Conv1d, HighwayLayer, ChainConcat, \
    DropoutLayer
from docqa.nn.recurrent_layers import CudnnLstm
from docqa.nn.similarity_layers import TriLinear
from docqa.nn.span_prediction import BoundsPredictor
from docqa.squad.build_squad_dataset import SquadCorpus
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.utils import get_output_name_from_cli


def main():
    """
    A close-as-possible impelemntation of BiDaF, its based on the `dev` tensorflow 1.1 branch of Ming's repo
    which, in particular, uses Adam not Adadelta. I was not able to replicate the results in paper using Adadelta,
    but with Adam i was able to get to 78.0 F1 on the dev set with this scripts. I believe this approach is
    an exact reproduction up the code in the repo, up to initializations.

    Notes: Exponential Moving Average is very important, as is early stopping. This is also in particualr best run
    on a GPU due to the large number of parameters and batch size involved.
    """
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=12, ema=0.999, async_encoding=10,
                               log_period=30, eval_period=1000, save_period=1000,
                               eval_samples=dict(dev=None, train=8000))

    # recurrent_layer = BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))
    # recurrent_layer = FusedLstm()
    recurrent_layer = SequenceMapperSeq(DropoutLayer(0.8), CudnnLstm(100))

    model = Attention(
        encoder=DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            embedder=LearnedCharEmbedder(16, 49, 8),
            layer=ReduceLayer("max", Conv1d(100, 5, 0.8), mask=False),
            shared_parameters=True
        ),
        word_embed_layer=None,
        embed_mapper=SequenceMapperSeq(
            HighwayLayer(activation="relu"), HighwayLayer(activation="relu"),
            recurrent_layer),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=BiAttention(TriLinear(bias=True), True),
        match_encoder=NullMapper(),
        predictor= BoundsPredictor(
            ChainConcat(
                start_layer=SequenceMapperSeq(
                    recurrent_layer,
                    recurrent_layer),
                end_layer=recurrent_layer
            )
        )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    eval = [LossEvaluator(), SquadSpanEvaluator(), BoundedSquadSpanEvaluator([18]), SentenceSpanEvaluator()]

    corpus = SquadCorpus()
    train_batching = ClusteredBatcher(60, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, False)
    data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)

    trainer.start_training(data, model, train_params, eval, model_dir.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()