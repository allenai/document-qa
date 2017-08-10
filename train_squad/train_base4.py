import runner
from data_processing.qa_data import FixedParagraphQaTrainingData
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder
from evaluator import LossEvaluator
from nn.attention import StaticAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import NullBiMapper, SequenceMapperSeq, FullyConnectedMerge
from nn.prediction_layers import ChainPredictor
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec, RecurrentEncoder, EncodeOverTime
from nn.similarity_layers import DotProduct
from runner import SerializableOptimizer, TrainParams
from squad.squad import SquadCorpus
from utils import get_output_name_from_cli

"""
Notes:

Drop 0.9 is worse, much worse on dev and a little worse on train, it looks like 
the train results will be better then drop 0.8 given time

Drop 0.7 is nearly as good or better than drop 0.8, but it learns slower. Noticeably worse on train,
i.e. the benifit is less overfitting as expected

We can drop down to size 70 with minimal perf. cost, and to size 60 with a minor cost on ev 
and slightly less overfitting

840 word vecs are nearly as good, IF you use batch 30

Were not able to get num/name placeholders to show a very strong benefit, we speculate this is because 
names/nums are not a big problem anyway.

Batch 30 reduce overfitting a bit, but at the cost of train score. overall slighly worse

100d match layers creates overfitting

Removing embedding doprout causes us to peak at 0.71 at then overfit

TODO:
try: (droout/rebuild layer, then deep LSTM w/o droput, then another droput-rebuild layer)
 -> partial dropout here might make sense
dropout between encoder and attention
batch 30 w/840d and drop 0.7, we speculate larger vecs will maek drop 0.7 less costly on train
(at some point) EMA, makes a big difference for BiDaF

Dropout ONLY the network
"""


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=16, log_period=20, eval_period=1400, save_period=1400,
                               max_dev_eval_examples=7000, max_train_eval_examples=7000)

    model = Attention(
        encoder=DocumentAndQuestionEncoder(),
        word_embed_layer=None,
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=15, init_scale=0.1),
            EncodeOverTime(RecurrentEncoder(LstmCellSpec(40), 'h'), mask=True),
            shared_parameters=True
        ),
        embed_mapper=None,
        question_mapper=SequenceMapperSeq(BiRecurrentMapper(LstmCellSpec(80, keep_probs=0.8))),
        context_mapper=SequenceMapperSeq(BiRecurrentMapper(LstmCellSpec(80, keep_probs=0.8))),
        memory_builder=NullBiMapper(),
        attention=StaticAttention(DotProduct(True), FullyConnectedMerge(160)),
        match_encoder=BiRecurrentMapper(LstmCellSpec(80, keep_probs=0.8)),
        predictor=ChainPredictor(
            start_layer=BiRecurrentMapper(LstmCellSpec(80, keep_probs=0.8)),
            end_layer=BiRecurrentMapper(LstmCellSpec(80, keep_probs=0.8))
        )
    )
    with open(__file__, "r") as f:
        notes = f.read()

    corpus = SquadCorpus()
    params = BatchingParameters(45, 45, "bucket_context_words_3",
                                "context_words", True, False)
    data = FixedParagraphQaTrainingData(corpus, None, params, [])

    eval = [LossEvaluator(), BoundedSpanEvaluator(bound=[17]), SentenceSpanEvaluator()]
    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, True)

if __name__ == "__main__":
    main()