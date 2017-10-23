import argparse

import numpy as np
import pandas as pd

from docqa.utils import print_table


"""
Compute our scores as more paragraphs are used from a 
"""


def compute_model_scores(df, max_over, target_score, group_cols):
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


def show_scores_table(df, n_to_show, cols):
    rows = [["Rank"] + cols]
    n_to_show = min(n_to_show, len(df))
    for i in range(n_to_show):
        rows.append(["%d" % (i+1)] + ["%.4f" % df[k].iloc[i] for k in cols])
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answers', help='answer file', nargs="+")
    parser.add_argument('--open', action="store_true")
    args = parser.parse_args()

    print("Loading answers..")
    answer_dfs = []
    for filename in args.answers:
        answer_dfs.append(pd.read_csv(filename))

    data = {}
    for i, answer_df in enumerate(answer_dfs):
        answer_df.sort_values(["question_id", "rank"], inplace=True)
        model_scores = compute_model_scores(answer_df, "predicted_score", "text_f1",
                                            ["question_id"] if args.open else ["question_id", "doc_id"])
        data["answers_%d_f1" % i] = model_scores
        model_scores = compute_model_scores(answer_df, "predicted_score", "text_em",
                                            ["question_id"] if args.open else ["question_id", "doc_id"])
        data["answers_%d_em" % i] = model_scores

    show_scores_table(pd.DataFrame(data), 30 if args.open else 15,
                      sorted(data.keys(), key=lambda x: (0, x) if "f1" in x else (1, x)))


if __name__ == "__main__":
    main()


