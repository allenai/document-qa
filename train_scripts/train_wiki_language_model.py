import trainer
from language_model import LanguageTrainingData, LmBatchingParameters, PartialFillIn, LanguageModelEncoder
from nn.embedder import FixedWordEmbedder
from nn.layers import SequenceMapperSeq, FullyConnected
from nn.recurrent_layers import BiRecurrentMapper, LstmCellSpec
from trainer import TrainParams, SerializableOptimizer
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    with open(__file__, "r") as f:
        notes = f.read()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=20, log_period=20, eval_period=1200, save_period=1200,
                               max_dev_eval_examples=7000, max_train_eval_examples=7000)

    params = LmBatchingParameters(45, 45, 3, 1, True, False)
    data = LanguageTrainingData(params,
                                TriviaQaEvidenceCorpusTxt(type="wikipedia"),
                                min_doc_len=15, sample_documents=10000, intern=True)
    model = PartialFillIn(
        LanguageModelEncoder(800),
        FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        SequenceMapperSeq(BiRecurrentMapper(LstmCellSpec(150)), FullyConnected(300, activation=None)),
        fill_in_percent=0.9, loss_fn="l1")

    trainer.start_training(data, model, train_params, [], trainer.ModelDir(out), notes, False)

if __name__ == "__main__":
    main()