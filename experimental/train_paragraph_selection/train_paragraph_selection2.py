from data_processing.qa_data import Batcher
from paragraph_selection.paragraph_selection_evaluators import AnyTopNEvaluator, TotalAnswersEvaluator, \
    PercentAnswerEvaluator
from paragraph_selection.paragraph_selection_featurizer import NGramMatchingFeaturizer, \
    ParagraphOrderFeatures, ParagraphFeatures

import trainer
from data_processing.document_splitter import MergeParagraphs
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.text_utils import WordNormalizer, NltkPlusStopWords
from evaluator import LossEvaluator
from experimental.paragraph_selection.paragraph_selection_model import NParagraphsSortKey, \
    ParagraphSelectionFeaturizer, SoftmaxPrediction, EncodedFeatures, SelectionDatasetBuilder
from nn.embedder import FixedWordEmbedder
from nn.layers import SequenceMapperSeq, FullyConnected, DropoutLayer, NullMapper, ConcatLayer
from nn.recurrent_layers import BiRecurrentMapper, GruCellSpec, RecurrentEncoder
from trainer import TrainParams, SerializableOptimizer
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import get_output_name_from_cli


def main():
    out = get_output_name_from_cli()

    train_params = TrainParams(SerializableOptimizer("Adadelta", dict(learning_rate=1.0)),
                               max_checkpoints_to_keep=1,
                               num_epochs=25, log_period=40, eval_period=1800, save_period=1800,
                               eval_samples=dict(dev=None, train=10000))

    stop = NltkPlusStopWords(True)
    norm = WordNormalizer(stemmer="wordnet")
    fe = ParagraphSelectionFeaturizer(MergeParagraphs(400), None,
                                                [
                                                    NGramMatchingFeaturizer(stop, norm, (1, 1)),
                                                 ],
                                                [ParagraphOrderFeatures(), ParagraphFeatures()],
                                                filter_initial_zeros=False,
                                                prune_no_answer=True)

    model = EncodedFeatures(
        word_embed=FixedWordEmbedder(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False),
        char_embed=None,
        featurizer=fe,
        map_question=SequenceMapperSeq(DropoutLayer(0.8),
                                       BiRecurrentMapper(GruCellSpec(80)),
                                       DropoutLayer(0.8),
                                       FullyConnected(30, activation="tanh")),
        merge_with_features=ConcatLayer(),
        map_joint=NullMapper(),
        encode_word_features=RecurrentEncoder(GruCellSpec(25), None),
        process=SequenceMapperSeq(BiRecurrentMapper(GruCellSpec(25)), FullyConnected(10)),
        predictor=SoftmaxPrediction(),
        any_features=True
    )

    with open(__file__, "r") as f:
        notes = f.read()

    train_batching = Batcher(45, NParagraphsSortKey(), True, False)
    eval_batching = Batcher(45, NParagraphsSortKey(), False, False)
    data = PreprocessedData(
        TriviaQaWebDataset(), fe,
        SelectionDatasetBuilder(train_batching, eval_batching),
        eval_on_verified=False,
        hold_out_train=(0, 6000),
        # sample=200, sample_dev=200,
    )

    # data.preprocess(8, chunk_size=1000)
    # data.cache_preprocess("unigram-para-held-out.pkl")
    data.load_preprocess("unigram-para-held-out.pkl")

    eval = [LossEvaluator(),
            AnyTopNEvaluator([1, 2, 3, 4]),
            PercentAnswerEvaluator([1, 2, 3, 4]),
            TotalAnswersEvaluator([1, 2, 3, 4])]
    trainer.start_training(data, model, train_params, eval, trainer.ModelDir(out), notes, False)


if __name__ == "__main__":
    main()