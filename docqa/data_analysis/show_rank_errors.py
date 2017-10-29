import argparse
from typing import List

import pandas as pd
import sys

from docqa.data_processing.document_splitter import ExtractedParagraphWithAnswers
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.squad.squad_data import SquadCorpus
from docqa.text_preprocessor import WithIndicators
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWebDataset
from docqa.utils import flatten_iterable
import numpy as np


def extract_paragraph(text: List[List[List[str]]], start, end) -> List[List[str]]:
    out = []
    on_token = 0
    on_para = []
    for para in text:
        for sent in para:
            exected_len = max(on_token - start, 0)
            if (sum(len(s) for s in out) + len(on_para)) != exected_len:
                raise ValueError()
            if on_token + len(sent) <= start:
                on_token += len(sent)
                continue
            if (on_token + len(sent)) >= end:
                on_para += sent[:end - on_token]
                out.append(on_para)
                if len(flatten_iterable(out)) != end - start:
                    raise ValueError(len(flatten_iterable(out)), end - start)
                return out
            if on_token + len(sent) < start:
                pass
            on_para += sent
            on_token += len(sent)
        if len(on_para) > 0:
            out.append(on_para)
            on_para = []

    out.append(on_para)
    if len(flatten_iterable(out)) != end - start:
        raise ValueError(len(flatten_iterable(out)), end-start)
    return out


stop = NltkPlusStopWords(True).words

class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    CYAN = "\033[96m"
    ENDC = '\033[0m'


def display_para(text: List[str], answers, question, p_start, p_end):
    words = {w.lower() for w in question if w.lower() not in stop}
    text = list(text)
    if answers is not None:
        for s,e in answers:
            text[s] = bcolors.CORRECT + text[s]
            text[e] = text[e] + bcolors.ENDC

    for i, word in enumerate(text):
        if word.lower() in words:
            text[i] = bcolors.ERROR + text[i] + bcolors.ENDC

    text[p_start] = bcolors.CYAN  + text[p_start]
    text[p_end] = text[p_end] + bcolors.ENDC

    return text


def show_squad_errors(answers):
    print("Loading answers..")
    answer_df = pd.read_csv(answers)

    print("Loading questions..")
    corpus = SquadCorpus()
    questions = {}
    docs = {}
    for doc in corpus.get_dev():
        for para in doc.paragraphs:
            for q in para.questions:
                questions[q.question_id] = q
                docs[q.question_id] = doc

    answer_df.sort_values(["question_id", "rank"], inplace=True)
    grouped = list(answer_df.groupby(["question_id"]))
    np.random.shuffle(grouped)

    for question_id, group in grouped:
        q = questions[question_id]
        doc = docs[question_id]
        cur_best_score = group.text_f1.iloc[0]
        cur_best_conf = group.predicted_score.iloc[0]
        cur_best_ix = group.index[0]
        for i in range(1, len(group)):
            ix = group.index[i]
            conf = group.predicted_score[ix]
            if conf > cur_best_conf:
                score = group.text_f1[ix]
                if score < cur_best_score:
                    # We hurt our selves!
                    print("Oh no!")
                    print(" ".join(q.words))
                    print(q.answer.answer_text)
                    print("Best score was %.4f (conf=%.4f), but not is %.4f (conf=%.4f)" % (
                        cur_best_score, cur_best_conf, score, conf
                    ))
                    cur_para = doc.paragraphs[group.para_number[cur_best_ix]]
                    new_para = doc.paragraphs[group.para_number[ix]]

                    p1_s, p1_e = group.predicted_start[cur_best_ix], group.predicted_end[cur_best_ix]
                    p2_s, p2_e = group.predicted_start[ix], group.predicted_end[ix]

                    print(" ".join(display_para(cur_para.get_context(), None, q.words, p1_s, p1_e)))
                    print()
                    print(" ".join(display_para(new_para.get_context(), None, q.words, p2_s, p2_e)))
                    input()
                else:
                    cur_best_score = score
                    cur_best_ix = ix
                    cur_best_conf = conf


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answers', help='answer file')
    parser.add_argument('question_source')
    args = parser.parse_args()

    print("Loading answers..")
    answer_df = pd.read_csv(args.answers)

    print("Loading questions..")
    if args.question_source == "open":
        corpus = TriviaQaOpenDataset()
        questions = {q.question_id: q for q in corpus.get_dev()}
    elif args.question_source == "web":
        corpus = TriviaQaWebDataset()
        questions = {}
        for q in corpus.get_dev():
            for d in q.all_docs:
                questions[(q.question_id, d.doc_id)] = q
    elif args.question_source == "squad":
        show_squad_errors(args.answers)
        return
    else:
        raise ValueError()

    pre = WithIndicators()

    answer_df.sort_values(["question_id", "rank"], inplace=True)

    if args.question_source == "open":
        iter = answer_df.groupby(["question_id"])
    else:
        iter = answer_df.groupby(["question_id", "doc_id"])

    grouped = list(iter)
    np.random.shuffle(grouped)

    for key, group in grouped:
        print(list(questions.keys())[:10])
        q = questions[key]
        cur_best_score = group.text_f1.iloc[0]
        cur_best_conf = group.predicted_score.iloc[0]
        cur_best_ix = group.index[0]
        for i in range(1, len(group)):
            ix = group.index[i]
            conf = group.predicted_score[ix]
            if conf > cur_best_conf:
                score = group.text_f1[ix]
                if score < cur_best_score:
                    # We hurt our selves!
                    print("Oh no!")
                    print(" ".join(q.question))
                    print(q.answer.all_answers)
                    print("Best score was %.4f (conf=%.4f), but not is %.4f (conf=%.4f)" % (
                        cur_best_score, cur_best_conf, score, conf
                    ))
                    d1 = [d for d in q.all_docs if d.doc_id == group.doc_id[cur_best_ix]][0]
                    p1 = extract_paragraph(corpus.evidence.get_document(d1.doc_id), group.para_start[cur_best_ix], group.para_end[cur_best_ix])
                    s, e = group.para_start[cur_best_ix], group.para_end[cur_best_ix]
                    answers = d1.answer_spans[np.logical_and(d1.answer_spans[:, 0] >= s, d1.answer_spans[:, 1] < s)] - s
                    p1 = pre.encode_extracted_paragraph(q.question, ExtractedParagraphWithAnswers(
                        p1,  group.para_start[cur_best_ix], group.para_end[cur_best_ix], answers))

                    d2 = [d for d in q.all_docs if d.doc_id == group.doc_id[ix]][0]
                    p2 = extract_paragraph(corpus.evidence.get_document(d2.doc_id), group.para_start[ix], group.para_end[ix])
                    s, e = group.para_start[ix], group.para_end[ix]
                    answers = d2.answer_spans[np.logical_and(d2.answer_spans[:, 0] >= s, d2.answer_spans[:, 1] < s)] - s
                    p2 = pre.encode_extracted_paragraph(q.question, ExtractedParagraphWithAnswers(
                        p2,  group.para_start[ix], group.para_end[ix], answers))

                    p1_s, p1_e = group.predicted_start[cur_best_ix], group.predicted_end[cur_best_ix]
                    p2_s, p2_e = group.predicted_start[ix], group.predicted_end[ix]
                    print(" ".join(display_para(p1.text, p1.answer_spans, q.question, p1_s, p1_e)))
                    print()
                    print(" ".join(display_para(p2.text, p2.answer_spans, q.question, p2_s, p2_e)))
                    input()
                else:
                    cur_best_score = score
                    cur_best_ix = ix
                    cur_best_conf = conf


if __name__ == "__main__":
    show_squad_errors(sys.argv[1])


