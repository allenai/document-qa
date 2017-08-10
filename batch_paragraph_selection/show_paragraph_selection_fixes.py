import argparse
import pickle

import matplotlib.pylab as plt
import numpy as np

from data_processing.qa_with_selection_data import ParagraphRanks
from data_processing.span_data import compute_span_f1
from squad.squad import SquadCorpus
from squad.squad_official_evaluation import f1_score as squad_official_f1_score


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("answers")
    parser.add_argument("paragraph")
    args = parser.parse_args()

    with open(args.answers, "rb") as f:
        answers = pickle.load(f)
    answers = {x.question_id: x for x in answers}

    para_predictions = ParagraphRanks(args.paragraph).get_ranks()

    docs = SquadCorpus().get_dev_docs()

    max_para_len = max(len(doc.paragraphs) for doc in docs)
    top_n_f1_score = np.zeros(max_para_len)
    counts = np.zeros(max_para_len)
    top_n_span_score = np.zeros(max_para_len)

    n_questions = 0
    for doc in docs:
        for para in doc.paragraphs:
            n_questions += len(para.questions)
            for question in para.questions:
                answer = answers[question.question_id]

                best_val = -1
                text_f1 = -1
                span_f1 = 0
                for r, i in enumerate(np.argsort(-np.array(para_predictions[question.question_id]))):
                    val = answer.span_vals[i]
                    if val > best_val:
                        best_val = val

                        answer_text = doc.paragraphs[i].get_original_text(*answer.spans[i])
                        text_f1 = 0
                        for ans in question.answer:
                            text_f1 = max(text_f1, squad_official_f1_score(answer_text, ans.text))

                        span_f1 = 0
                        if i == para.paragraph_num:  # correct paragraph
                            for ans in question.answer:
                                span_f1 = max(span_f1, compute_span_f1(answer.spans[i], (ans.para_word_start, ans.para_word_end)))

                    top_n_f1_score[r] += text_f1
                    top_n_span_score[r] += span_f1

                top_n_f1_score[len(doc.paragraphs):max_para_len] += text_f1
                top_n_span_score[len(doc.paragraphs):max_para_len] += text_f1

    plt.plot(np.arange(0, max_para_len)+1, top_n_f1_score/n_questions)
    plt.show()


if __name__ == "__main__":
    main()


