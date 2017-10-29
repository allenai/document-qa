import argparse

import numpy as np
import pandas as pd

from docqa.utils import print_table


def compute_model_scores(df, max_over, target_score, group_cols):
    scores = []
    for _, group in df.groupby(group_cols):
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


def show_scores_table(scores, n_to_show=10):
    rows = [["Rank"] + ["Score"]]
    n_to_show = min(n_to_show, len(scores))
    for i in range(n_to_show):
        rows.append(["%d" % (i+1)] + ["%.4f" % scores.iloc[i]])
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answers', help='answer files', nargs="+")
    args = parser.parse_args()

    print("Loading answers..")
    answer_dfs = []
    for f in args.answers:
        answer_dfs.append(pd.read_csv(f))

    print("Scoring...")
    out = {}
    for f, answer_df in zip(args.answers, answer_dfs):
        answer_df.sort_values(["rank"], inplace=True)
        model_scores = compute_model_scores(answer_df, "predicted_score", "text_f1", ["question_id"])
        out["f1-" + f] = model_scores
        model_scores = compute_model_scores(answer_df, "predicted_score", "text_em", ["question_id"])
        out["em-" + f] = model_scores

    print(pd.DataFrame(out))


if __name__ == "__main__":
    main()


