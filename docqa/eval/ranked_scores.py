import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

from docqa.utils import print_table


def compute_ranked_scores(df, max_over, target_score, group_cols):
    scores = []
    for _, group in df[[max_over, target_score] + group_cols].groupby(group_cols):
        if target_score == max_over:
            scores.append(group[target_score].cummax().values)
        else:
            used_predictions = group[max_over].expanding().apply(lambda x: x.argmax())
            scores.append(group[target_score].iloc[used_predictions].values)

    max_para = max(len(x) for x in scores)
    summed_scores = np.zeros(max_para)
    for s in scores:
        summed_scores[:len(s)] += s
        summed_scores[len(s):] += s[-1]
    return summed_scores/len(scores)


def show_scores_table(df, cols):
    rows = [["Rank"] + cols]
    for i in range(len(df)):
        rows.append(["%d" % (i+1)] + ["%.4f" % df[k].iloc[i] for k in cols])
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description=
                                     "Compute scores as more paragraphs are used, using "
                                     "a per-paragraph csv file as built from our evaluation scripts ")
    parser.add_argument('answers', help='answer file(s)', nargs="+")
    parser.add_argument('--per_doc', action="store_true",
                        help="Show scores treating each (quesiton, document) pair as a "
                             "datapoint, instead of each question. Should be used for the TriviaQA Web"
                             " dataset")
    args = parser.parse_args()

    print("Loading answers..")
    answer_dfs = []
    for filename in args.answers:
        answer_dfs.append(pd.read_csv(filename))

    print("Computing ranks...")
    if args.per_doc:
        group_by = ["question_id", "doc_id"]
    else:
        group_by = ["question_id"]

    data = OrderedDict()
    for i, answer_df in enumerate(answer_dfs):
        answer_df.sort_values(["question_id", "rank"], inplace=True)
        model_scores = compute_ranked_scores(answer_df, "predicted_score", "text_em", group_by)
        data["answers_%d_em" % i] = model_scores
        model_scores = compute_ranked_scores(answer_df, "predicted_score", "text_f1", group_by)
        data["answers_%d_f1" % i] = model_scores

    show_scores_table(pd.DataFrame(data),
                      sorted(data.keys(), key=lambda x: (0, x) if x.endswith("em") else (1, x)))


if __name__ == "__main__":
    main()


