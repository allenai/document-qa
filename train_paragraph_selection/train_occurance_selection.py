import trainer
from data_processing.document_splitter import MergeParagraphs
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import WordNormalizer, NltkPlusStopWords
from evaluator import LossEvaluator
from nn.attention import AttentionEncoder
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import SequenceMapperSeq, FullyConnected, DropoutLayer, MultiAggregateLayer, MergeWith, SelfProduct, \
    NullMapper, ConcatLayer, FullyConnectedMerge, ConcatWithProductProj, ConcatOneSidedProduct
from nn.recurrent_layers import BiRecurrentMapper, GruCellSpec, RecurrentEncoder, EncodeOverTime, RecurrentMapper
from paragraph_selection.paragraph_selection_evaluators import AnyTopNEvaluator
from paragraph_selection.paragraph_selection_featurizer import NGramMatchingFeaturizer, \
    ParagraphOrderFeatures, ParagraphFeatures, NGramFineGrained
from paragraph_selection.paragraph_selection_model import NParagraphsSortKey, \
    ParagraphSelectionFeaturizer, WeightedFeatures, SoftmaxPrediction, FeaturersOnly, SigmoidPredictions, \
    EncodedFeatures, SelectionDatasetBuilder
from paragraph_selection.paragraph_selection_with_context import SelectionWithContextDatasetBuilder, ContextTriAttention
from paragraph_selection.word_occurance_model import OccuranceFeaturizer, EncodedOccurancePredictor, \
    OccuranceDatasetBuilder
from trainer import TrainParams, SerializableOptimizer
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               max_checkpoints_to_keep=1,
                               num_epochs=25, log_period=40, eval_period=1800, save_period=1800,
                               eval_samples=dict(dev=None, train=8000))

    stop = NltkPlusStopWords(True)
    norm = WordNormalizer(stemmer="wordnet")
    fe = OccuranceFeaturizer(MergeParagraphs(400), None,
                             [ParagraphOrderFeatures(), ParagraphFeatures()],
                             stop, norm, True, False)


    model = EncodedOccurancePredictor(
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        char_embed=None,
        featurizer=fe,
        # question_map=SequenceMapperSeq(DropoutLayer(0.8), BiRecurrentMapper(GruCellSpec(80)), DropoutLayer(0.8), FullyConnected(40, activation="tanh")),
        question_map=FullyConnected(30, activation="tanh"),
        occurance_encoder=RecurrentEncoder(GruCellSpec(25), None),
        paragraph_encoder=SequenceMapperSeq(BiRecurrentMapper(GruCellSpec(20)), FullyConnected(10, activation="tanh")),
        prediction_layer=SoftmaxPrediction(),
        feature_vec_size=5,
        distance_vecs=20
    )

    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = Batcher(30, NParagraphsSortKey(), True, False)
    eval_batching = Batcher(30, NParagraphsSortKey(), False, False)
    data = PreprocessedData(
        TriviaQaWebDataset(), fe,
        OccuranceDatasetBuilder(train_batching, eval_batching),
        eval_on_verified=False,
        # sample=30000, sample_dev=5000,
    )

    # data.preprocess(8, chunk_size=1000)
    # data.cache_preprocess("occurance-data.pkl")
    data.load_preprocess("occurance-data.pkl")

    eval = [LossEvaluator(), AnyTopNEvaluator([1, 2, 3, 4])]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()