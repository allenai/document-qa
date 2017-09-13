import argparse
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from data_processing.text_utils import NltkPlusStopWords
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from utils import flatten_iterable, print_table



def compute_cumsum(df, target_score, by_doc=True):
    scores = []
    if by_doc:
        group_cols = ["quid", "doc_id"]
    else:
        group_cols = ["quid"]
    for quid, group in df.groupby(group_cols):
        scores.append(group[target_score].cummax().values)

    max_para = max(len(x) for x in scores)
    summed_scores = np.zeros(max_para)
    for s in scores:
        summed_scores[:len(s)] += s
        summed_scores[len(s):] += s[-1]
    return summed_scores/len(scores)


def compute_oracle_scores(df):
    """ Compute document level or multi-document level scores from paragraph-level scores stored in a
    DataFrame. The DataFrame should be sorted so that paragraphs (per question) are sorted in the
     would be should be considered as input to the model.
    """
    scores = []
    oracle_scores = []
    answer_paragraph_oracle = []
    model_paragraph_oracle = []

    for quid, group in df.groupby(["quid", "doc_id"]):
        used_predictions = group["predicted_score"].expanding().apply(lambda x: x.argmax())
        scores.append(group["text_f1"].iloc[used_predictions].values)

        answer_sorted = group.sort_values("n_answers", ascending=False)
        used_predictions = answer_sorted["predicted_score"].expanding().apply(lambda x: x.argmax())
        answer_paragraph_oracle.append(answer_sorted["text_f1"].iloc[used_predictions].values)

        oracle_scores.append(group["oracle_f1"].cummax().values)
        model_paragraph_oracle.append(group["text_f1"].cummax().values)

    max_para = max(len(x) for x in scores)

    summed_scores = np.zeros(max_para)
    summed_oracle = np.zeros(max_para)
    summed_model_paragraph_oracle = np.zeros(max_para)
    summed_answer_paragraph_oracle = np.zeros(max_para)
    total_counts = np.zeros(max_para, dtype=np.int64)

    for s in scores:
        summed_scores[:len(s)] += s
        summed_scores[len(s):] += s[-1]
        total_counts[:len(s)] += 1
    for s in oracle_scores:
        summed_oracle[:len(s)] += s
        summed_oracle[len(s):] += s[-1]
    for s in answer_paragraph_oracle:
        summed_answer_paragraph_oracle[:len(s)] += s
        summed_answer_paragraph_oracle[len(s):] += s[-1]
    for s in model_paragraph_oracle:
        summed_model_paragraph_oracle[:len(s)] += s
        summed_model_paragraph_oracle[len(s):] += s[-1]

    l = len(oracle_scores)
    return pd.DataFrame(dict(oracle=summed_oracle/l,
                        model=summed_scores/l,
                        counts=total_counts,
                        paragraph_oracle=summed_model_paragraph_oracle/l,
                        answer_oracle=summed_answer_paragraph_oracle/l))


def show_scores_table(df, n_to_show=10):
    cols = ['model']
    rows = [["Rank"] + cols]
    n_to_show = min(n_to_show, len(df))
    for i in range(n_to_show):
        rows.append(["%d" % (i+1)] + ["%.4f" % df[k].iloc[i] for k in cols])
    print_table(rows)


def show_scores(df):
    para_to_show = 30
    x = np.arange(0, para_to_show)
    plt.plot(x, df.oracle[:para_to_show], label="oracle")
    plt.plot(x, df.model[:para_to_show], label='model')
    plt.plot(x, df.answer_oracle[:para_to_show], label="answer_oracle")
    plt.plot(x, df.paragraph_oracle[:para_to_show], label="paragraph_oracle")
    plt.legend()
    plt.show()


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


def print_questions(question, answers, context, answer_span):
    print(" ".join(question))
    print(answers)
    context = flatten_iterable(context)
    for s,e in answer_span:
        context[s] = "{{{" + context[s]
        context[e] = context[e] + "}}}"
    print(" ".join(context))


def show_reorder_examples(df, evidence, questions):
    stop = NltkPlusStopWords(punctuation=True).words
    q_map = {q.question_id:q for q in questions}
    score = 0
    total = 0
    n_q = 0
    for (quid, doc_id), group in df.groupby(["quid", "doc_id"]):
        max_ix = group["text_f1"].argmax()
        q = q_map[quid]
        question_words = set(x.lower() for x in q.question if x.lower() not in stop)
        text = None
        n_q += 1
        # total += len(group)

        found_any = False
        for ix in range(len(group)):
            if group["para_start"].iloc[ix] == 0:
                continue
            # if text is None:
            #     text = evidence.get_document(doc_id, flat=True)
            # s = group.para_start.iloc[ix]
            # e = group.para_end.iloc[ix]
            # paragraph = text[s:e + 1]
            # if not any(x.lower() in question_words for x in paragraph):
            #     continue
            total += 1
            if group["text_f1"].iloc[ix] <= group.text_f1.iloc[0]:
                continue
            score += 1
            # break

            # argmax_text = text[group.predicted_start[max_ix]:group.predicted_end[max_ix] + 1]
            # if any(x.lower() in question_words for x in argmax_text):
            #     continue
            # reorder += 1
            # print(" ".join(q.question))
            # print(doc_id)
            # print(q.answer.all_answers)
            # for ix in [group.index[0], max_ix]:
            #     s = group.predicted_start[ix]
            #     e = group.predicted_end[ix]
            #     start = group.para_start[ix]
            #     end = group.para_end[ix]
            #     group_text = text[start:end+1]
            #     group_text[s] = bcolors.CORRECT + group_text[s]
            #     group_text[e+1] = group_text[e+1] + bcolors.ENDC
            #     print(" ".join(group_text))
            #     print()
            # # input()
            # print()
    print(n_q, total)
    print(total/n_q)
    print(score, total, score/total)


def show_missing(df, evidence, questions):
    stop = NltkPlusStopWords(punctuation=True).words
    q_map = {q.question_id:q for q in questions}
    total = 0
    for (quid, doc_id), group in df.groupby(["quid", "doc_id"]):
        total += 1
        q = q_map[quid]
        question_words = set(x.lower() for x in q.question if x.lower() not in stop)
        spans = [x for x in q.all_docs if x.doc_id == doc_id][0].answer_spans
        text = None
        for ix in range(len(group)):
            if group["text_f1"].iloc[ix] > 0.5 or group["n_answers"].iloc[ix] == 0:
                continue
            if text is None:
                text = evidence.get_document(doc_id, flat=True)
            s = group.para_start.iloc[ix]
            e = group.para_end.iloc[ix]
            paragraph = text[s:e + 1]
            if not any(x.lower() in question_words for x in paragraph):
                continue
            para_spans = spans[np.logical_and(spans[:, 0] >= s, spans[:, 1] < e)] - s
            print(" ".join(q.question))
            for s,e in para_spans:
                paragraph[s] = bcolors.CORRECT + paragraph[s]
                paragraph[e] = paragraph[e] + bcolors.ENDC
            print(" ".join(paragraph))
            print()
            input()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('evaluation_filename')
    parser.add_argument('--doc', action="store_true")
    args = parser.parse_args()

    print("Load...")
    df = pd.read_csv(args.evaluation_filename)
    print("Sort...")
    df.sort_values(["quid", "para_start", "para_end"], inplace=True)
    print("Score...")
    score_df = compute_oracle_scores(df)
    # show_scores(score_df)
    show_scores_table(score_df)
    # print("Showing...")
    # show_missing(df, corpus.evidence, corpus.get_dev())
    # show_reorder_examples(df, corpus.evidence, corpus.get_dev())


if __name__ == "__main__":
    main()
