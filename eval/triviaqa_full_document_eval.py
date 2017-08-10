import argparse
import json
import pickle
from os.path import join
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import trainer
from data_processing.document_splitter import DocumentSplitter, MergeParagraphs, ParagraphFilter, ContainsQuestionWord, \
    TopTfIdf
from data_processing.qa_data import ParagraphAndQuestionDataset, NoShuffleBatcher
from data_processing.text_utils import NltkPlusStopWords
from evaluator import Evaluator, Evaluation
from model import ModelOutput
from trainer import ModelDir
from trivia_qa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import TriviaQaQuestion
from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
from trivia_qa.triviaqa_training_data import TriviaQaAnswer, DocumentParagraphQuestion
from utils import ResourceLoader, flatten_iterable


class RecordParagraphSpanPrediction(Evaluator):
    """ Computes the best span in tensorflow, meaning which we expect to be faster and
    does not require us having to keep the entire set of output logits in RAM """
    def __init__(self, bound: int):
        self.bound = bound

    def tensors_needed(self, model):
        span, score = model.prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[DocumentParagraphQuestion], true_len, **kargs):
        print("Begining evaluation")
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        oracle_f1s = np.zeros(len(data))

        print("Scoring...")
        for i in tqdm(range(len(data)), total=len(data)):
            point = data[i]
            if point.answer is None:
                continue
            text = flatten_iterable(point.context)
            pred_span = spans[i]
            pred_text = " ".join(text[pred_span[0]:pred_span[1] + 1])
            f1 = 0
            for answer in data[i].answer.answer_aliases:
                f1 = max(f1, trivia_f1_score(pred_text, answer))
            pred_f1s[i] = f1

            oracle_f1 = 0
            for s,e in point.answer.answer_spans:
                pred_text = " ".join(text[s:e+1])
                for answer in point.answer.answer_aliases:
                    oracle_f1 = max(oracle_f1, trivia_f1_score(pred_text, answer))
            oracle_f1s[i] = oracle_f1

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["text_f1"] = pred_f1s
        results["oracle_f1"] = oracle_f1s
        results["para_start"] = [x.para_range[0] for x in data]
        results["para_end"] = [x.para_range[1] for x in data]
        results["quid"] = [x.q_id for x in data]
        results["doc_id"] = [x.doc_id for x in data]
        return Evaluation({}, results)


class RecordParagraphResult(Evaluator):
    def tensors_needed(self, prediction: ModelOutput):
        super().tensors_needed(prediction)


def eval(model_dir: ModelDir, batch_size: int,
         test_questions: List[TriviaQaQuestion],
         document_splitter: DocumentSplitter,
         paragraph_filter: Optional[ParagraphFilter],
         corpus: TriviaQaEvidenceCorpusTxt,
         n_questions: Optional[int], output: Optional[str]=None, ema=False):

    if n_questions is not None:
        test_questions.sort(key=lambda x:x.question_id)
        np.random.RandomState(0).shuffle(test_questions)
        test_questions = test_questions[:n_questions]

    print("Loading docs...")
    questions = []
    n_pairs = 0
    n_filtered = 0
    for q in tqdm(test_questions):
        for doc in q.all_docs:
            n_pairs += 1
            text = corpus.get_document(doc.doc_id, document_splitter.reads_first_n)
            if text is None:
                raise ValueError()
            paragraphs = document_splitter.split(text, doc.answer_spans)
            unfiltered_len = len(paragraphs)
            if paragraph_filter is not None:
                paragraphs = paragraph_filter.prune(q.question, paragraphs)
                n_filtered += unfiltered_len - len(paragraphs)

            for para in paragraphs:
                answer = TriviaQaAnswer(para.answer_spans, q.answer.all_answers)
                questions.append(DocumentParagraphQuestion(q.question_id, doc.doc_id, (para.start, para.end),
                                                           q.question, para.text, answer))

    print("Split %d q-doc pairs into %d q-paragraph pairs (%d filtered)" % (n_pairs, len(questions), n_filtered))
    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(questions, key=lambda x: (len(x.context), len(x.question)), reverse=True)

    print("Done, starting eval")

    checkpoint = model_dir.get_latest_checkpoint()
    data = ParagraphAndQuestionDataset(questions, NoShuffleBatcher(batch_size, True))

    model = model_dir.get_model()
    evaluation = runner.test(model,
                             [RecordParagraphSpanPrediction(8)],
                             [data], ResourceLoader(), checkpoint, ema, None)[0]

    print("Saving result")
    output_file = join(model_dir.get_eval_dir(), output)

    if output_file.endswith("json"):
        with open(output_file, "w") as f:
            json.dump(evaluation.per_sample, f)
    elif output_file.endswith("pkl"):
        with open(output_file, "wb") as f:
            pickle.dump(evaluation.per_sample, f)
    elif output_file.endswith("csv"):
        import pandas as pd
        df = pd.DataFrame(evaluation.per_sample)
        df.to_csv(output_file, index=False)
    else:
        raise ValueError("Unrecognized file format")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument('output', type=str)
    parser.add_argument('-n', '--n_sample', type=int, default=None)
    parser.add_argument('-s', '--splitter', type=str, default="merge-400",
                        choices=["merge-400", "merge-800"])
    parser.add_argument('-f', '--filter', type=str, default="none",
                        choices=["contains-question-word", "tfidf", "none"])
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus',
                        choices=["web-dev", "web-verified-dev", "web-train",
                                 "open-dev", "open-train"],
                        default="web-verified-dev")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)

    if args.corpus.startswith('web'):
        dataset = TriviaQaWebDataset()
        corpus = dataset.evidence
        if args.corpus == "web-dev":
            data = dataset.get_dev()
        elif args.corpus == "web-verified-dev":
            data = dataset.get_verified()
        elif args.corpus == "web-train":
            data = dataset.get_train()
        else:
            raise RuntimeError()
    else:
        dataset = TriviaQaOpenDataset()
        corpus = dataset.evidence
        if args.corpus == "open-dev":
            data = dataset.get_dev()
        elif args.corpus == "open-train":
            data = dataset.get_train()
        else:
            raise RuntimeError()

    if args.splitter == "merge-400":
        splitter = MergeParagraphs(400)
    else:
        splitter = MergeParagraphs(800)

    if args.filter == "contains-question-word":
        para_filter = ContainsQuestionWord(NltkPlusStopWords(punctuation=True))
    elif args.filter == "tfidf":
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), 6)
    else:
        para_filter = None

    eval(model_dir, args.get_fixed_batch_size, data, splitter, para_filter,
         corpus, args.n_sample, args.output, True)


def tmp():
    with open("/tmp/test.pkl", "rb") as f:
        data = pickle.load(f)
        print(data)

if __name__ == "__main__":
    main()
    # tmp()




