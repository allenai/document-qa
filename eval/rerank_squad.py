import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_processing.text_utils import NltkPlusStopWords
from squad.document_level import SquadTfIdfRanker
from squad.squad_data import SquadCorpus
from utils import flatten_iterable, print_table


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


def show_scores_table(df, n_to_show=10):
    cols = ['model']
    rows = [["Rank"] + cols]
    n_to_show = min(n_to_show, len(df))
    for i in range(n_to_show):
        rows.append(["%d" % (i+1)] + ["%.4f" % df[k].iloc[i] for k in cols])
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answers', help='answer file')
    parser.add_argument('-r', '--rank', type=str, default="none", choices=["tfidf", "first"])
    parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--ema', action="store_true")
    args = parser.parse_args()

    print("Loading answers..")
    answer_df = pd.read_csv(args.answers)

    print("Loading questions..")
    corpus = SquadCorpus()
    docs = corpus.get_dev() if args.corpus.endswith("dev") else corpus.get_train()

    print("Computing ranks..")
    ranker = SquadTfIdfRanker(NltkPlusStopWords(), 1000, False)
    ranks = {}
    for doc in tqdm(docs):
        scores = ranker.rank(doc)
        all_questions = flatten_iterable(x.questions for x in doc.paragraphs)
        for q_ix, q in enumerate(all_questions):
            for p_ix, p in enumerate(doc.paragraphs):
                ranks[(q.question_id, p.paragraph_num)] = scores[q_ix, p_ix]

    ranks_col = []
    for quid, num in answer_df[["quid", "para_number"]].itertuples(index=False):
        ranks_col.append(ranks[(quid, num)])
    answer_df["ranks"] = ranks_col

    print("Scoring...")
    answer_df.sort_values(["ranks"], inplace=True)
    model_scores = compute_model_scores(answer_df, "predicted_score", "text_f1", ["quid"])
    show_scores_table(pd.DataFrame(), 10)


if __name__ == "__main__":
    main()


