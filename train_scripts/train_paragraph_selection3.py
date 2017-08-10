import runner
from data_processing.document_splitter import MergeParagraphs
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import WordNormalizer, NltkPlusStopWords
from evaluator import LossEvaluator
from nn.attention import AttentionEncoder
from nn.embedder import FixedWordEmbedder, CharWordEmbedder, LearnedCharEmbedder
from nn.layers import SequenceMapperSeq, FullyConnected, DropoutLayer, MultiAggregateLayer, MergeWith, SelfProduct, \
    NullMapper, ConcatLayer, FullyConnectedMerge, ConcatWithProductProj, ConcatOneSidedProduct
from nn.recurrent_layers import BiRecurrentMapper, GruCellSpec, RecurrentEncoder, EncodeOverTime
from paragraph_selection.paragraph_selection_evaluators import AnyTopNEvaluator
from paragraph_selection.paragraph_selection_featurizer import NGramMatchingFeaturizer, \
    ParagraphOrderFeatures, ParagraphFeatures, NGramFineGrained
from paragraph_selection.paragraph_selection_model import NParagraphsSortKey, \
    ParagraphSelectionFeaturizer, WeightedFeatures, SoftmaxPrediction, FeaturersOnly, SigmoidPredictions, \
    EncodedFeatures, SelectionDatasetBuilder
from paragraph_selection.paragraph_selection_with_context import SelectionWithContextDatasetBuilder, ContextTriAttention
from runner import TrainParams, SerializableOptimizer
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
    fe = ParagraphSelectionFeaturizer(MergeParagraphs(400), None,
                                      [
                                          NGramMatchingFeaturizer(stop, norm, (1, 1)),
                                          # NGramFineGrained(stop, norm, [(1, (True, True, True)),
                                          #                               (2, (True, True, True))]),
                                      ],
                                      [ParagraphOrderFeatures(), ParagraphFeatures()],
                                      filter_initial_zeros=False,
                                      prune_no_answer=True,
                                      context_voc=True)

    model = ContextTriAttention(
        word_embed=FixedWordEmbedder(vec_name="glove.6B.100d", word_vec_init_scale=0, learn_unk=False),
        featurizer=fe,
        map_atten=FullyConnected(20, activation="tanh"),
        map_question=SequenceMapperSeq(
            DropoutLayer(0.8),
            BiRecurrentMapper(GruCellSpec(80)),
            DropoutLayer(0.8), FullyConnected(30, activation="tanh")),
        merge_with_features=ConcatLayer(),
        # map_joint=BiRecurrentMapper(GruCellSpec(20)),
        map_joint=NullMapper(),
        # map_joint=SequenceMapperSeq(BiRecurrentMapper(GruCellSpec(20)), DropoutLayer(0.8)),
        encode_word_features=RecurrentEncoder(GruCellSpec(25), None),
        process=SequenceMapperSeq(BiRecurrentMapper(GruCellSpec(25)), FullyConnected(10)),
        predictor=SoftmaxPrediction(),
        any_features=True
    )

    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = Batcher(30, NParagraphsSortKey(), True, False)
    eval_batching = Batcher(30, NParagraphsSortKey(), False, False)
    data = PreprocessedData(
        TriviaQaWebDataset(), fe,
        SelectionWithContextDatasetBuilder(train_batching, eval_batching),
        eval_on_verified=False,
        # sample=200, sample_dev=200,
    )

    # data.preprocess(8, chunk_size=1000)
    # data.cache_preprocess("unigram-para.pkl")
    data.load_preprocess("unigram-para.pkl")

    eval = [LossEvaluator(), AnyTopNEvaluator([1, 2, 3, 4])]
    runner.start_training(data, model, train_params, eval, runner.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()