import argparse
import numpy as np
from tqdm import tqdm

from data_processing.document_splitter import MergeParagraphs, TopTfIdf, ContainsQuestionWord, Truncate, FirstN
from data_processing.preprocessed_corpus import preprocess_par
from data_processing.text_utils import NltkPlusStopWords
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.training_data import ExtractMultiParagraphsPerQuestion, ExtractMultiParagraphs
from trivia_qa.trivia_qa_eval import normalize_answer


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--n_sample', type=int, default=None)
    parser.add_argument('-s', '--splitter', type=str, default="merge", choices=["merge", "truncate"])
    parser.add_argument('-t', '--n_tokens', type=int, default=400)
    parser.add_argument('-p', '--n_paragraphs', type=int, default=1)
    parser.add_argument('-f', '--filter', type=str, default="tfidf")
    parser.add_argument('-c', '--corpus', type=str, default="dev", choices=["train", "dev"])
    args = parser.parse_args()

    dataset = TriviaQaWebDataset()
    if args.splitter == "merge":
        splitter = MergeParagraphs(args.n_tokens)
    else:
        splitter = Truncate(args.n_tokens)

    if args.filter == "contains-question-word":
        para_filter = ContainsQuestionWord(NltkPlusStopWords(punctuation=True), n_paragraphs=args.n_paragraphs)
    elif args.filter == "tfidf":
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), args.n_paragraphs)
    elif args.filter == "first":
        para_filter = FirstN(args.n_paragraphs)
    else:
        raise NotImplementedError()

    if args.corpus == "dev":
        questions = dataset.get_dev()
    elif args.corpus == "train":
        questions = dataset.get_train()

    if args.n_sample is not None:
        questions = np.random.choice(questions, args.n_sample, False)

    prep = ExtractMultiParagraphs(splitter, para_filter, None, require_an_answer=False)
    prepped_data = preprocess_par(questions, dataset.evidence, prep, 6, 1000)

    n_found = 0
    n_answer_span = 0
    n_total = 0
    for q in tqdm(prepped_data.data, ncols=80):
        for i, p in enumerate(q.paragraphs):
            if len(p.answer_spans) > 0:
                n_answer_span += 1
            n_total += 1

    print("%d/%d (%.4f)" % (n_answer_span, n_total, n_answer_span / n_total))
    print("%d/%d (%.4f)" % (n_found, n_total, n_found/n_total))


if __name__ == "__main__":
    main()
