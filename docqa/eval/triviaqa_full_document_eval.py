import argparse
import json
from os.path import join
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from docqa import trainer
from docqa.config import TRIVIA_QA
from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, FirstN
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import FixedOrderBatcher
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.evaluator import Evaluator, Evaluation
from docqa.model_dir import ModelDir
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset, TriviaQaOpenDataset, TriviaQaWikiDataset
from docqa.triviaqa.read_data import normalize_wiki_filename
from docqa.triviaqa.training_data import DocumentParagraphQuestion, ExtractMultiParagraphs, \
    ExtractMultiParagraphsPerQuestion
from docqa.triviaqa.trivia_qa_eval import exact_match_score as trivia_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as trivia_f1_score
from docqa.utils import ResourceLoader, print_table

"""
Evaluate on TriviaQA data
"""


class RecordParagraphSpanPrediction(Evaluator):

    def __init__(self, bound: int, record_text_ans: bool):
        self.bound = bound
        self.record_text_ans = record_text_ans

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        needed = dict(spans=span, model_scores=score)
        return needed

    def evaluate(self, data: List[DocumentParagraphQuestion], true_len, **kargs):
        spans, model_scores = np.array(kargs["spans"]), np.array(kargs["model_scores"])

        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        text_answers = []

        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring"):
            point = data[i]
            if point.answer is None and not self.record_text_ans:
                continue
            text = point.get_context()
            pred_span = spans[i]
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
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on TriviaQA data')
    parser.add_argument('model', help='model directory')
    parser.add_argument('-p', '--paragraph_output', type=str,
                        help="Save fine grained results for each paragraph in csv format")
    parser.add_argument('-o', '--official_output', type=str, help="Build an offical output file with the model's"
                                                                  " most confident span for each (question, doc) pair")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--n_processes', type=int, default=None,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-i', '--step', type=int, default=None, help="checkpoint to load, default to latest")
    parser.add_argument('-n', '--n_sample', type=int, default=None, help="Number of questions to evaluate on")
    parser.add_argument('-a', '--async', type=int, default=10)
    parser.add_argument('-t', '--tokens', type=int, default=400,
                        help="Max tokens per a paragraph")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=15,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-f', '--filter', type=str, default=None, choices=["tfidf", "truncate", "linear"],
                        help="How to select paragraphs")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes might be faster but wll take more memory")
    parser.add_argument('--max_answer_len', type=int, default=8,
                        help="Max answer span to select")
    parser.add_argument('-c', '--corpus',
                        choices=["web-dev", "web-test", "web-verified-dev", "web-train",
                                 "open-dev", "open-train", "wiki-dev", "wiki-test"],
                        default="web-verified-dev")
    args = parser.parse_args()

    model_dir = ModelDir(args.model)
    model = model_dir.get_model()

    if args.corpus.startswith('web'):
        dataset = TriviaQaWebDataset()
        if args.corpus == "web-dev":
            test_questions = dataset.get_dev()
        elif args.corpus == "web-test":
            test_questions = dataset.get_test()
        elif args.corpus == "web-verified-dev":
            test_questions = dataset.get_verified()
        elif args.corpus == "web-train":
            test_questions = dataset.get_train()
        else:
            raise AssertionError()
    elif args.corpus.startswith("wiki"):
        dataset = TriviaQaWikiDataset()
        if args.corpus == "wiki-dev":
            test_questions = dataset.get_dev()
        elif args.corpus == "wiki-test":
            test_questions = dataset.get_test()
        else:
            raise AssertionError()
    else:
        dataset = TriviaQaOpenDataset()
        if args.corpus == "open-dev":
            test_questions = dataset.get_dev()
        elif args.corpus == "open-train":
            test_questions = dataset.get_train()
        else:
            raise AssertionError()

    corpus = dataset.evidence
    splitter = MergeParagraphs(args.tokens)

    per_document = args.corpus.startswith("web")  # wiki and web are both multi-document

    filter_name = args.filter
    if filter_name is None:
        # Pick default depending on the kind of data we are using
        if per_document:
            filter_name = "tfidf"
        else:
            filter_name = "linear"

    print("Selecting %d paragraphs using method \"%s\" per %s" % (
        args.n_paragraphs, filter_name, ("question-document pair" if per_document else "question")))

    if filter_name == "tfidf":
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), args.n_paragraphs)
    elif filter_name == "truncate":
        para_filter = FirstN(args.n_paragraphs)
    elif filter_name == "linear":
        para_filter = ShallowOpenWebRanker(args.n_paragraphs)
    else:
        raise ValueError()

    n_questions = args.n_sample
    if n_questions is not None:
        test_questions.sort(key=lambda x:x.question_id)
        np.random.RandomState(0).shuffle(test_questions)
        test_questions = test_questions[:n_questions]

    print("Building question/paragraph pairs...")
    # Loads the relevant questions/documents, selects the right paragraphs, and runs the model's preprocessor
    if per_document:
        prep = ExtractMultiParagraphs(splitter, para_filter, model.preprocessor, require_an_answer=False)
    else:
        prep = ExtractMultiParagraphsPerQuestion(splitter, para_filter, model.preprocessor, require_an_answer=False)
    prepped_data = preprocess_par(test_questions, corpus, prep, args.n_processes, 1000)

    data = []
    for q in prepped_data.data:
        for i, p in enumerate(q.paragraphs):
            if q.answer_text is None:
                ans = None
            else:
                ans = TokenSpans(q.answer_text, p.answer_spans)
            data.append(DocumentParagraphQuestion(q.question_id, p.doc_id,
                                                 (p.start, p.end), q.question, p.text,
                                                  ans, i))

    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    print("Done, starting eval")

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    test_questions = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluation = trainer.test(model,
                             [RecordParagraphSpanPrediction(args.max_answer_len, True)],
                              {args.corpus:test_questions}, ResourceLoader(), checkpoint, not args.no_ema, args.async)[args.corpus]

    if not all(len(x) == len(data) for x in evaluation.per_sample.values()):
        raise RuntimeError()

    df = pd.DataFrame(evaluation.per_sample)

    if args.official_output is not None:
        print("Saving question result")

        fns = {}
        if per_document:
            # I didn't store the unormalized filenames exactly, so unfortunately we have to reload
            # the source data to get exact filename to output an official test script
            print("Loading proper filenames")
            if args.corpus == 'web-test':
                source = join(TRIVIA_QA, "qa", "web-test-without-answers.json")
            elif args.corpus == "web-dev":
                source = join(TRIVIA_QA, "qa", "web-dev.json")
            else:
                raise AssertionError()

            with open(join(source)) as f:
                data = json.load(f)["Data"]
            for point in data:
                for doc in point["EntityPages"]:
                    filename = doc["Filename"]
                    fn = join("wikipedia", filename[:filename.rfind(".")])
                    fn = normalize_wiki_filename(fn)
                    fns[(point["QuestionId"], fn)] = filename

        answers = {}
        scores = {}
        for q_id, doc_id, start, end, txt, score in df[["question_id", "doc_id", "para_start", "para_end",
                                                        "text_answer", "predicted_score"]].itertuples(index=False):
            filename = dataset.evidence.file_id_map[doc_id]
            if per_document:
                if filename.startswith("web"):
                    true_name = filename[4:] + ".txt"
                else:
                    true_name = fns[(q_id, filename)]
                key = q_id + "--" + true_name
            else:
                key = q_id

            prev_score = scores.get(key)
            if prev_score is None or prev_score < score:
                scores[key] = score
                answers[key] = txt

        with open(args.official_output, "w") as f:
            json.dump(answers, f)

    output_file = args.paragraph_output
    if output_file is not None:
        print("Saving paragraph result")
        df.to_csv(output_file, index=False)

    print("Computing scores")

    if per_document:
        group_by = ["question_id", "doc_id"]
    else:
        group_by = ["question_id"]

    # Print a table of scores as more paragraphs are used
    df.sort_values(group_by + ["rank"], inplace=True)
    f1 = compute_ranked_scores(df, "predicted_score", "text_f1", group_by)
    em = compute_ranked_scores(df, "predicted_score", "text_em", group_by)
    table = [["N Paragraphs", "EM", "F1"]]
    table += list([str(i+1), "%.4f" % e, "%.4f" % f] for i, (e, f) in enumerate(zip(em, f1)))
    print_table(table)
if __name__ == "__main__":
    main()




