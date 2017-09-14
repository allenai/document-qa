import argparse
import json
import pickle
from os.path import join
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import trainer
from data_processing.document_splitter import DocumentSplitter, MergeParagraphs, ParagraphFilter, ContainsQuestionWord, \
    TopTfIdf, ShallowOpenWebRanker
from data_processing.preprocessed_corpus import preprocess_par
from data_processing.qa_training_data import ParagraphAndQuestionDataset
from data_processing.span_data import TokenSpans
from data_processing.text_utils import NltkPlusStopWords
from dataset import FixedOrderBatcher
from evaluator import Evaluator, Evaluation
from model_dir import ModelDir
from trivia_qa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from trivia_qa.read_data import TriviaQaQuestion
from trivia_qa.training_data import DocumentParagraphQuestion, ExtractMultiParagraphs, ExtractMultiParagraphsPerQuestion
from trivia_qa.trivia_qa_eval import f1_score as trivia_f1_score
from trivia_qa.trivia_qa_eval import exact_match_score as trivia_em_score
from utils import ResourceLoader, flatten_iterable


class RecordParagraphSpanPrediction(Evaluator):
    """ Computes the best span in tensorflow, meaning which we expect to be faster and
    does not require us having to keep the entire set of output logits in RAM """
    def __init__(self, bound: int, record_text_ans: bool,
                 use_prob: bool):
        self.bound = bound
        self.use_prob = use_prob
        self.record_text_ans = record_text_ans

    def tensors_needed(self, prediction):
        if self.use_prob:
            span, score = prediction.get_best_span_prob(self.bound)
        else:
            span, score = prediction.get_best_span(self.bound)
        needed = dict(spans=span, model_scores=score)
        if hasattr(prediction, "none_logit"):
            needed["none_logit"] = prediction.none_logit
            needed["none_prob"] = prediction.none_prob
        return needed

    def evaluate(self, data: List[DocumentParagraphQuestion], true_len, **kargs):
        print("Begining evaluation")
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []

        print("Scoring...")
        for i in tqdm(range(len(data)), total=len(data), ncols=80):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            text = point.get_context()
            pred_span = spans[i]
            # print(pred_span)
            pred_text = " ".join(text[pred_span[0]:pred_span[1] + 1])
            if self.record_text_ans:
                text_answers.append(pred_text)
                if point.answer is None:
                    continue

            f1 = 0
            em = False
            for answer in data[i].answer.answer_text:
                f1 = max(f1, trivia_f1_score(pred_text, answer))
                if not em:
                    em = trivia_em_score(pred_text, answer)

            pred_f1s[i] = f1
            pred_em[i] = em

        results = {}
        results["n_answers"] = [0 if x.answer is None else len(x.answer.answer_spans) for x in data]
        if self.record_text_ans:
            results["text_answer"] = text_answers
        results["predicted_score"] = model_scores
        results["predicted_start"] = spans[:, 0]
        results["predicted_end"] = spans[:, 1]
        results["text_f1"] = pred_f1s
        results["rank"] = [x.rank for x in data]
        results["text_em"] = pred_em
        results["para_start"] = [x.para_range[0] for x in data]
        results["para_end"] = [x.para_range[1] for x in data]
        results["question_id"] = [x.question_id for x in data]
        results["doc_id"] = [x.doc_id for x in data]
        if "none_logit" in kargs:
            results["none_logit"] = kargs["none_logit"]
            results["none_prob"] = kargs["none_prob"]
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='name of output to exmaine')
    parser.add_argument('-p', '--paragraph_output', type=str, help="Save fine grained results for each paragraph")
    parser.add_argument('-q', '--question_predictions', type=str, help="Save predictions for each question")
    parser.add_argument('-d', '--question_doc_predictions', type=str, help="Save predictions for each question-doc pair")
    parser.add_argument('-e', '--ema', action="store_true")
    parser.add_argument('-r', '--prob', action="store_true")
    parser.add_argument('-i', '--step', type=int, default=None)
    parser.add_argument('-n', '--n_sample', type=int, default=None)
    parser.add_argument('-a', '--async', type=int, default=10)
    parser.add_argument('-s', '--splitter', type=str, default="merge-400",
                        choices=["merge-400", "merge-800"])
    parser.add_argument('-f', '--filter', type=str, default="tfidf-6")
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-c', '--corpus',
                        choices=["web-dev", "web-verified-dev", "web-train",
                                 "open-dev", "open-train"],
                        default="web-verified-dev")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    model = model_dir.get_model()

    if args.corpus.startswith('web'):
        dataset = TriviaQaWebDataset()
        corpus = dataset.evidence
        if args.corpus == "web-dev":
            test_questions = dataset.get_dev()
        elif args.corpus == "web-verified-dev":
            test_questions = dataset.get_verified()
        elif args.corpus == "web-train":
            test_questions = dataset.get_train()
        else:
            raise RuntimeError()
    else:
        dataset = TriviaQaOpenDataset()
        corpus = dataset.evidence
        if args.corpus == "open-dev":
            test_questions = dataset.get_dev()
        elif args.corpus == "open-train":
            test_questions = dataset.get_train()
        else:
            raise RuntimeError()

    if args.splitter == "merge-400":
        splitter = MergeParagraphs(400)
    else:
        splitter = MergeParagraphs(800)

    per_document = True
    if args.filter == "contains-question-word":
        para_filter = ContainsQuestionWord(NltkPlusStopWords(punctuation=True))
    elif args.filter.startswith("tfidf-"):
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), int(args.filter[6:]))
    elif args.filter.startswith("truncate-"):
        para_filter = None
    elif args.filter.startswith("q-rank-"):
        para_filter = ShallowOpenWebRanker(int(args.filter[7:]))
        per_document = False
    elif args.filter == "none":
        para_filter = None
    else:
        raise ValueError()

    n_questions = args.n_sample
    if n_questions is not None:
        test_questions.sort(key=lambda x:x.question_id)
        np.random.RandomState(0).shuffle(test_questions)
        test_questions = test_questions[:n_questions]

    print("Building question/paragraph pairs...")
    if per_document:
        prep = ExtractMultiParagraphs(splitter, para_filter, model.preprocessor, require_an_answer=False)
    else:
        prep = ExtractMultiParagraphsPerQuestion(splitter, para_filter, model.preprocessor, require_an_answer=False)

    prepped_data = preprocess_par(test_questions, corpus, prep, 6, 1000)

    data = []
    for q in prepped_data.data:
        for i, p in enumerate(q.paragraphs):
            data.append(DocumentParagraphQuestion(q.question_id, p.doc_id,
                                                 (p.start, p.end), q.question, p.text,
                                                  TokenSpans(q.answer_text, p.answer_spans),
                                                  i))

    n_filtered = len(test_questions) - prepped_data.true_len

    print("Split %d q-doc pairs into %d q-paragraph pairs (%d filtered)" % (
        sum(len(x.all_docs) for x in test_questions), len(data), n_filtered))

    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    print("Done, starting eval")

    if args.step is None:
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        checkpoint = model_dir.get_checkpoint(args.step)

    test_questions = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluation = trainer.test(model,
                             [RecordParagraphSpanPrediction(8, True, args.prob)],
                              {args.corpus:test_questions}, ResourceLoader(), checkpoint, args.ema, args.async)[args.corpus]

    if not all(len(x) == len(data) for x in evaluation.per_sample.values()):
        raise RuntimeError()

    import pandas as pd
    df = pd.DataFrame(evaluation.per_sample)
    df.sort_values(["question_id", "doc_id", "predicted_score"], inplace=True, ascending=False)

    q_pred = df.groupby(["question_id"]).first()
    print("Question EM: %.4f" % q_pred["text_em"].mean())
    print("Question F1: %.4f" % q_pred["text_f1"].mean())

    doc_pred = df.groupby(["question_id", "doc_id"]).first()
    print("Doc-Question EM: %.4f" % doc_pred["text_em"].mean())
    print("Doc-Question F1: %.4f" % doc_pred["text_f1"].mean())

    print("Para-Question EM: %.4f" % df["text_em"].mean())
    print("Para-Question F1: %.4f" % df["text_f1"].mean())

    output_file = args.paragraph_output
    if output_file is not None:
        print("Saving paragraph result")
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

    # if args.question_predictions is not None:
    #     print("Saving question predictions")
    #     pred = {}
    #     for quid, group in df.groupby(["question_id"]):
    #         selected = group["predicted_score"].argmax()
    #         pred[quid] = group["text_answer"][selected]
    #     with open(args.question_predictions, "w") as f:
    #         json.dump(evaluation.per_sample, f)
    #
    # if args.question_doc_predictions is not None:
    #     print("Saving question-doc predictions")
    #     pred = {}
    #     for id, group in df.groupby(["question_id", "doc_id"]):
    #         selected = group["predicted_score"].argmax()
    #         pred[id] = group["text_answer"][selected]
    #     with open(args.question_doc_predictions, "w") as f:
    #         json.dump(evaluation.per_sample, f)

if __name__ == "__main__":
    main()




