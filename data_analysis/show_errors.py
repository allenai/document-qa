import argparse
import json
import numpy as np
from squad.squad_official_evaluation import exact_match_score

from data_processing.paragraph_qa import split_docs
from squad.build_squad_dataset import SquadCorpus


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("answer_file")
    parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    args = parser.parse_args()

    corpus = SquadCorpus()
    if args.corpus == "dev":
        questions = corpus.get_dev()
    else:
        questions = corpus.get_train()
    questions = split_docs(questions)

    with open(args.answer_file, "r") as f:
        answers = json.load(f)

    wrong = 0
    np.random.shuffle(questions)
    for q in questions:
        answer = answers[q.question_id]
        if all(exact_match_score(answer, a) == 0 for a in q.answer.answer_text):
            print(answer + " ||| " + str(list(set(q.answer.answer_text))))
            wrong += 1

    print(wrong / len(questions))


if __name__ == "__main__":
    main()