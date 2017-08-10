import argparse
import json
from typing import List, Optional
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import trainer
from data_processing.document_splitter import TopTfIdf, MergeParagraphs
from data_processing.preprocessed_corpus import PreprocessedData, preprocess_par
from data_processing.qa_data import ParagraphAndQuestion, ParagraphAndQuestionDataset, NoShuffleBatcher, Batcher
from data_processing.text_utils import NltkPlusStopWords
from dataset import ListBatcher, ClusteredBatcher
from evaluator import Evaluator, Evaluation, RecordSpanPrediction, RecordQuestionId, LossEvaluator
from paragraph_selection.paragraph_selection_evaluators import AnyTopNEvaluator
from paragraph_selection.paragraph_selection_model import NParagraphsSortKey, SelectionDatasetBuilder, \
    FeaturizeredParagraph
from trainer import ModelDir
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
from trivia_qa.triviaqa_evaluators import TfTriviaQaBoundedSpanEvaluator
from trivia_qa.triviaqa_training_data import TriviaQaAnswer
from utils import ResourceLoader, flatten_iterable, transpose_lists, print_table, NumpyEncoder

"""
Script to build an official trivia-qa prediction file
"""


class RecordParagraphRanks(Evaluator):
    def __init__(self, top_n: Optional[int]=None):
        self.top_n = top_n

    def tensors_needed(self, model):
        scores = model.prediction.paragraph_scores
        if self.top_n is None:
            n = tf.shape(scores)[1]
        else:
            n = tf.minimum(tf.shape(scores)[1], self.top_n)
        values, indices = tf.nn.top_k(scores, k=n)
        return dict(values=values, indices=indices)

    def evaluate(self, data: List[FeaturizeredParagraph], true_len, **kwards):
        scores = kwards["values"]
        indices = kwards["indices"]
        output_spans = []
        output_scores = []
        for i in range(len(data)):
            point_indices = indices[i]
            point_scores = scores[i]
            n_para = data[i].n_paragraphs
            if n_para < len(scores[i]):
                point_indices = point_indices[:n_para]
                point_scores = point_scores[:n_para]
            output_spans.append(data[i].spans[point_indices])
            output_scores.append(point_scores)
        return Evaluation({}, dict(
            question_id=[x.question_id for x in data],
            doc_id=[x.doc_id for x in data],
            paragraph_scores=output_scores, paragraph_spans=output_spans))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--n_processes", type=int, default=1)
    parser.add_argument('-n', '--sample_questions', type=int, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus', choices=["web-train", "web-dev", "web-verified-dev"], default="web-dev")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    model = model_dir.get_model()

    if args.corpus == "web-dev" or args.corpus == "web-verified-dev" or args.corpus == "web-train":
        data = TriviaQaWebDataset()
        evidence = data.evidence
        print("Loading questions...")
        if args.corpus == "web-dev":
            trivia_questions = data.get_dev()
        elif args.corpus == "web-train":
            trivia_questions = data.get_train()
        elif args.corpus == "web-verified-dev":
            trivia_questions = data.get_verified()
        else:
            raise ValueError()
    else:
        raise ValueError()

    if args.sample_questions:
        trivia_questions.sort(key=lambda x: x.question_id)
        trivia_questions = np.random.RandomState(0).choice(trivia_questions, args.sample_questions, replace=False)

    print("Preprocessing")
    data = preprocess_par(trivia_questions, evidence, model.featurizer, args.n_processes, 1000)
    batching = ClusteredBatcher(args.get_fixed_batch_size, NParagraphsSortKey(), False, True)
    dataset = SelectionDatasetBuilder(batching, batching).build_dataset(data, evidence, False)

    evaluators = [LossEvaluator(), AnyTopNEvaluator([1, 2, 3, 4])]

    if args.output is not None:
        evaluators.append(RecordParagraphRanks())

    checkpoint = model_dir.get_latest_checkpoint()
    evaluation = runner.test(model_dir.get_model(), evaluators, [dataset], ResourceLoader(), checkpoint, False)[0]

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))

    if args.output is not None:
        q_ids = evaluation.per_sample["question_id"]
        doc_ids = evaluation.per_sample["doc_id"]
        scores = evaluation.per_sample["paragraph_scores"]
        spans = evaluation.per_sample["paragraph_spans"]
        if args.output.endswith(".json"):
            output = []
            for q_id, doc_id, para_scores, para_spans in zip(q_ids, doc_ids, scores, spans):
                output.append(dict(quid=q_id, doc_id=doc_id, spans=list(para_spans), scores=list(para_scores)))

            with open(args.output, "w") as f:
                json.dump(output, f, cls=NumpyEncoder)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main()
    # tmp()




