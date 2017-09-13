import model_dir
import trainer
from data_processing.document_splitter import MergeParagraphs, TopTfIdf
from data_processing.qa_training_data import ContextLenKey, ContextLenBucketedKey
from data_processing.text_utils import NltkPlusStopWords
from dataset import ListBatcher, ClusteredBatcher
from doc_qa_models import ContextOnly
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from evaluator import LossEvaluator
from nn.embedder import FixedWordEmbedder, DropNames
from nn.layers import NullBiMapper, FullyConnected
from nn.recurrent_layers import CudnnGru
from nn.span_prediction import BoundsPredictor
from squad.squad_data import SquadCorpus, DocumentQaTrainingData
from squad.squad_evaluators import BoundedSquadSpanEvaluator
from trainer import SerializableOptimizer, TrainParams
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               num_epochs=16, eval_period=900, log_period=30,
                               async_encoding=5,
                               save_period=900, eval_samples=dict(train=6000, dev=6000))

    model = ContextOnly(
        DocumentAndQuestionEncoder(SingleSpanAnswerEncoder()),
        FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        None,
        FullyConnected(50),
        BoundsPredictor(NullBiMapper())
    )

    corpus = SquadCorpus()
    train_batching = ClusteredBatcher(45, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(45, ContextLenKey(), False, False)
    data = DocumentQaTrainingData(corpus, None, train_batching, eval_batching)

    eval = [LossEvaluator(), BoundedSquadSpanEvaluator(bound=[17])]
    trainer.start_training(data, model, train_params, eval, model_dir.ModelDir(out), "")

if __name__ == "__main__":
    # tmp()
    main()