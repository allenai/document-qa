import argparse
from typing import List

import pandas as pd

from data_processing.document_splitter import ExtractedParagraph, ExtractedParagraphWithAnswers
from data_processing.text_utils import NltkPlusStopWords
from squad.squad_data import SquadCorpus
from text_preprocessor import WithIndicators
from trivia_qa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWebDataset
from utils import flatten_iterable
import numpy as np



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answers', help='answer file')
    args = parser.parse_args()

    print("Loading answers..")
    answer_df = pd.read_csv(args.answers)
    print(answer_df.columns)

    print("Loading questions..")
    corpus = SquadCorpus()
    para_to_len = {}
    for doc in corpus.get_dev():
        for p in doc.paragraphs:
            for q in p.questions:
                for para in doc.paragraphs:
                    para_to_len[(q.question_id, para.paragraph_num)] = para.n_tokens

    lens = [para_to_len[tuple(x)] for x in answer_df[["question_id", "para_number"]].itertuples(index=False)]

    answer_df["length"] = lens

    answer_df.to_csv("/tmp/tmp.csv")


if __name__ == "__main__":
    main()


