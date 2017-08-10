import trainer
from data_processing.span_data import SpanCorpus
from data_processing.text_data import QuestionFilter, AnswerWord, ParagraphAndQuestionTrainingData, BatchingParameters
from doc_qa_models import Attention
from encoder import DocumentAndQuestionEncoder
from evaluator import LossEvaluator, SquadSpanEvaluator
from nn.attention import StaticAttention
from nn.embedder import FixedWordEmbedder
from nn.layers import NullBiMapper, NullMapper, SequenceMapperSeq, FullyConnectedMerge
from nn.prediction_layers import ChainPredictor
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec
from nn.similarity_layers import DotProduct
from trainer import SerializableOptimizer, TrainParams
from utils import get_output_name_from_cli


def main():
    """
    Simple baseline, with out word information about 71, with word informatio about 74
    I have experimented a fair bit with this base, and eventually found the key point to improving on
    it is to shrink down the number of hidden units to manage overfitting
    """
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=16, log_period=20, eval_period=1100, save_period=1100,
                               max_dev_eval_examples=6000, max_train_eval_examples=6000)
    para_size_th = 400
    data_filters = [QuestionFilter(ques_size_th=32),
                    AnswerWord(para_size_th=para_size_th)]

    model = Attention(
        encoder=DocumentAndQuestionEncoder(para_size_th=para_size_th, all_answers=False),
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=None,
        embed_mapper=SequenceMapperSeq(BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))),
        question_mapper=None,
        context_mapper=None,
        memory_builder=NullBiMapper(),
        attention=StaticAttention(DotProduct(True), FullyConnectedMerge(200)),
        match_encoder=NullMapper(),
        predictor=ChainPredictor(
            start_layer=BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8)),
            end_layer=BiRecurrentMapper(LstmCellSpec(100, keep_probs=0.8))
        )
    )
    with open(__file__, "r") as f:
        notes = f.read()

    corpus = SpanCorpus("squad")
    params = BatchingParameters(45, 45, "bucket_context_words_3",
                                "context_words", True, False)
    data = ParagraphAndQuestionTrainingData(corpus, None, params, data_filters)

    eval = [LossEvaluator(), SquadSpanEvaluator(corpus, bound=[17])]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes, False)

if __name__ == "__main__":
    main()