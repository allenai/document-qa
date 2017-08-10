import runner
from data_processing.qa_data import BatchingParameters, FixedParagraphQaTrainingData
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder
from evaluator import LossEvaluator, SpanEvaluator, SentenceSpanEvaluator
from nn.attention import SelfAttention
from nn.attention_recurrent_layers import RecurrentAttention
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import SequenceMapperSeq, NullBiMapper, DropoutLayer
from nn.recurrent_layers import EncodeOverTime, RecurrentEncoder, LstmCellSpec, GruCellSpec, BiRecurrentMapper
from runner import TrainParams, SerializableOptimizer
from squad.squad import SquadCorpus
from utils import get_output_name_from_cli


def main():
    """ """
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adam", dict(learning_rate=0.001)),
                               num_epochs=12, ema=0.999,
                               log_period=30, eval_period=1000, save_period=1000,
                               max_dev_eval_examples=6000, max_train_eval_examples=6000)

    enc = SequenceMapperSeq(
        DropoutLayer(0.8),
        BiRecurrentMapper(GruCellSpec(75)),
        DropoutLayer(0.8),
        BiRecurrentMapper(GruCellSpec(75)),
        DropoutLayer(0.8),
        BiRecurrentMapper(GruCellSpec(75)),
        DropoutLayer(0.8)
    )

    # TODO still to figure out what the heck is going on with their predictor approach....
    model = Attention(
        encoder=DocumentAndQuestionEncoder(),
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        char_embed=CharWordEmbedder(  # its not specified how large the character dimension should be
            LearnedCharEmbedder(word_size_th=14, char_th=50, char_dim=25, init_scale=0.1),
            EncodeOverTime(RecurrentEncoder(GruCellSpec(75), 'h', 0.8, None), mask=True),
            shared_parameters=True
        ),
        word_embed_layer=None,
        embed_mapper=None,
        question_mapper=enc,
        context_mapper=enc,
        memory_builder=NullBiMapper(),
        attention=RecurrentAttention(GruCellSpec(75), direction="bidirectional", gated=True),
        match_encoder=SequenceMapperSeq(DropoutLayer(0.8),
                                        SelfAttention(RecurrentAttention(GruCellSpec(75), direction="bidirectional", gated=True)),
                                        DropoutLayer(0.8)),
        predictor= ChainConcatPredictor(
            start_layer=SequenceMapperSeq(
                BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8)),
                BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))),
            end_layer=BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))
        )
    )

    with open(__file__, "r") as f:
        notes = f.read()

    eval = [LossEvaluator(), SpanEvaluator(), SentenceSpanEvaluator()]

    corpus = SquadCorpus()
    params = BatchingParameters(60, 60, "bucket_context_words_3",
                                "context_words", True, False)
    data = FixedParagraphQaTrainingData(corpus, None, params, [])

    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()