import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.paragraph_qa import ContextLenKey, ContextLenBucketedKey
from data_processing.qa_data import Batcher, FixedParagraphQaTrainingData
from data_processing.text_utils import NltkPlusStopWords
from dataset import ListBatcher, ClusteredBatcher
from doc_qa_models import ContextOnly
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.embedder import FixedWordEmbedder
from nn.layers import NullBiMapper, FullyConnected
from nn.span_prediction import BoundsPredictor
from trainer import SerializableOptimizer, TrainParams
from squad.build_dataset import SquadCorpus
from squad.squad_eval import BoundedSquadSpanEvaluator, SentenceSpanEvaluator
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=16, eval_period=100, log_period=30,
                               async_encoding=5,
                               save_period=8000, eval_samples=dict(train=1000, dev=500))

    model = ContextOnly(
        DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        FixedWordEmbedder(vec_name="glove.6B.100d", learn_unk=False, word_vec_init_scale=0),
        None,
        FullyConnected(100),
        BoundsPredictor(NullBiMapper())
    )

    corpus = SquadCorpus()
    train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = FixedParagraphQaTrainingData(corpus, None, train_batching, eval_batching)

    eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17]), SentenceSpanEvaluator()]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), "", False)

if __name__ == "__main__":
    # tmp()
    main()